import dash
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime

from dash_iconify import DashIconify
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback, callback_context
from dash_bootstrap_components import Container
from pathlib import Path
from dash.exceptions import PreventUpdate
from datetime import  timedelta
from plotly.subplots import make_subplots

from root_path import *

dis2 = str(10)

####################### Data import #######################

with open(root_path + '/assets/station_meta.pkl', 'rb') as f:
    meta = pd.read_pickle(f)
with open(root_path + '/assets/hourly.pkl', 'rb') as f:
    hourly = pd.read_pickle(f)
with open(root_path + '/assets/daily.pkl', 'rb') as f:
    daily = pd.read_pickle(f)
daily['DATE'] = pd.to_datetime(daily['DATE'], format='%Y-%m-%d %H')

# define dropdown options
station_options = []
for i, row in meta[['STATIONSNAME', 'STATIONS_ID']].iterrows():
    station_options.extend([{'label': row['STATIONSNAME'], 'value': row['STATIONS_ID']}])

year_options = list(range(pd.to_datetime(daily.DATE).dt.year.min(), pd.to_datetime(daily.DATE).dt.year.max()+1))

dash.register_page(__name__,
                   path='/Hitzewellen',
                   name='Hitzewellen',  # name of page, commonly used as name of link
                   title='Hitzewellen',  # title that appears on browser's tab
                   description='Übersicht über DWD-Daten zur Identifikation von Hitzewellen',
                   icon="fa-solid fa-temperature-high", 
                   order=3
)


####################### Utility functions #######################

def divide_dates_into_sublists(dates):
    sublists = []
    sublist = [dates[0]]
    for i in range(1, len(dates)):
        current_date = dates[i]
        previous_date = dates[i - 1]
        
        if (current_date - previous_date) > timedelta(days=1):
            sublists.append(sublist)
            sublist = []
        sublist.append(current_date)
    sublists.append(sublist)
    return sublists

def convert_date_format(date_str):
    date_object = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    new_date_str = date_object.strftime("%d/%m/%Y")
    return new_date_str


####################### Text Elemente #######################

markdown_referenzen = '''
## Referenzen
Gasparrini, A. and Armstrong, B. (2011). The impact of heat waves on mortality. *Epidemiology*, 22(1):68.  
Huth, R., Kyselyy, J., and Pokorna, L. (2000). A GCM simulation of heat waves, dry spells, and their relationships to circulation. *Climatic Change*, 46(1-2):29–60.  
Kysely, J. (2004). Mortality and displaced mortality during heat waves in the Czech Republic. *International Journal of Biometeorology*, 49:91–97.  
Kysely, J. (2010). Recent severe heat waves in central Europe: How to view them in a long-term prospect? *International Journal of Climatology: A Journal of the Royal Meteorological Society*, 30(1):89–109.  
Meehl, G. A. and Tebaldi, C. (2004). More intense, more frequent, and longer lasting heat waves in the 21st century. *Science*, 305(5686):994–997.
'''

markdown_heatwaves = '''
# Hitzewellen

Der urbane Hitzeinseleffekt - auf der [Einführungsseite](/) ausführlich beschrieben - ist insbesondere dann problematisch, wenn extreme Temperaturen an aufeinanderfolgenden Tagen auftreten (Gasparrini und Armstrong, 2011).
Solche Hitzeperioden werden üblicherweise auch als Hitzewellen bezeichnet.
Wir haben daher unsere gesamte Analyse auf Temperaturdaten ausgerichtet, die zur Zeit einer Hitzewelle aufgezeichnet wurden.
Ganz konkret wurde für die Berechnung der mittäglichen Oberflächentemperatur ein Mittelwert über Tage berechnet, welche Teil einer Hitzewelle waren.
Weitere Informationen hierzu finden Sie auch auf [Seite 4](/LST).  
Fragt sich bloß noch wie genau eine Hitzeweelle definiert ist.
Wir sind hierbei der herkömmlichen Definition von Huth et al. (2000) gefolgt,
welche in der metereologischen Literatur für Mitteleuropa meistens verwendet wird (Meehl und Tebaldi, 2004; Kysely, 2004, 2010).
Sie lautet:
'''

citation = '''
<blockquote style='background-color:#f9f9f9; padding:10px; margin:1em;'>
Eine Hitzewelle wird dann festgestellt, sobald an mindestens drei aufeinanderfolgenden Tagen eine Temperatur von 30°C überschritten wird.
Die Hitzewelle dauert so lange an, wie die durchschnittliche Höchsttemperatur über den gesamte Zeitraum bei über 30°C bleibt und
an keinem einzelnen Tag unter eine Höchsttemperatur von 25°C fällt.
</blockquote>
'''

markdown_header = '''
## Hitzewellen in München auf Basis von Daten des Deutschen Wetterdienstes
'''

markdown_plots = '''
Für die Identifikation von Hitzewellen haben wir Daten des Deutschen Wetterdienstes (kurz: DWD) verwendet.
Die nebenstehende Karte lokalisiert die drei Wetterstationen des DWDs, die innerhalb unseres Untersuchungsgebiets liegen.
Hiervon liegt eine im Stadtzentrum Münches, eine in Oberhaching und eine in unmittelbarer Nähe des Flughafens.

Die darunterliegenden Grafiken sollen einen vertieften Einblick in die Temperaturen in München Stadt und Umland gewähren.
Es können hierbei immer zwei der drei Stationen ausgewählt werden.
Die Stationsauswahl bezieht sich dann auf beide Grafiken.  
Unter anderem soll hiermit auf die nicht unbeträchtlichen Temperatur-Unterschiede zwischen Stadt und Land aufmerksam gemacht werden.
Die linke Darstellung zeigt für einen ausgewählten Tag zwischen dem 01.01.2014 und dem 31.12.2022 die stündlichen Temperaturdaten (in °C) an.
Die rechte Grafik visualisiert eine Zeitreihe von Höchsttemperaturen.
Eine Hitzewelle (nach oben genannter Definition) wird durch einen orange gefärbten Hintergrund angezeigt.
Bei entsprechender Stationsauswahl wird sehr gut deutlich, dass in Perioden in denen eine Hitzewelle in der Stadt auftritt, nicht zwangsläufig auch eine im ländlichen Umfeld präsent sein muss.
'''


####################### Map Element ##########################

osm_attribution = ('Map: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>')

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles: &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')
dwd_attibution = (' | Stations: &copy; <a href="https://opendata.dwd.de/">DWD Open Data</a>')

map_element = dl.Map(
    [
    dl.LayersControl( 
        [dl.BaseLayer(dl.TileLayer(attribution = osm_attribution + dwd_attibution), name = 'OpenStreetMap', checked=True)] +
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = osm_attribution + esri_attribution + dwd_attibution), name = 'ESRI Satellite')] 
    )],
    center=[48.1632, 11.5429],
    style={'width': '100%', 'height':'100%', 'margin': "auto", "display": "block"},
    zoom=9)

for index, row in meta.iterrows():

    latitude = row['GEOBREITE']
    longitude = row['GEOLAENGE']
    
    marker = dl.Marker(position=[latitude, longitude], children=[
        dl.Tooltip(html.Div([
            str(row['STATIONSNAME']),
            html.Br(),
            html.B('Aktiv seit '), convert_date_format(str(row['VON_DATUM'])),
            html.Br(),
            html.B('Höhe: '), ' ' + str(row['STATIONSHOEHE']), html.B('m')
        ]))
    ])
    
    map_element.children.append(marker)


####################### Layout #######################

layout = dbc.Container(
    [
        html.Div(style={'height': dis2 + 'vh'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(markdown_heatwaves, style={"text-align": "justify"}, dangerously_allow_html=True),
                        dcc.Markdown(citation, style={"text-align": "justify"}, dangerously_allow_html=True),
                        html.Br()
                    ],
                    className="mt-4",
                    )
            ],
            #className="mb-4"
        ),
        dbc.Row(
            [
                dcc.Markdown(markdown_header, dangerously_allow_html=True),
                dbc.Col(
                    [
                        dcc.Markdown(markdown_plots, style={"text-align": "justify"})
                    ],
                    #className="mt-4",  
                    width = 6
                ),
                dbc.Col(
                    [
                        map_element
                    ],
                    #className="mt-4",  
                    width = 6
                )
            ],
            className="mb-4"
        ),
        html.Hr(),
        dbc.Row(
            [   
                dbc.Col(
                    [
                        html.P('Bitte Stationen für Vergleich wählen:', style={'font-size': '20px'})
                    ],
                    width = 4
                ),
                dbc.Col(
                    [   
                        html.Label('Station 1', style={'font-weight': 'bold'}),
                        dcc.Dropdown(
                            id='station_1',
                            options=station_options,
                            value=station_options[1]['value'],
                            clearable=False,
                            style={'border-color':'red','border-width':'3px'}
                        ),
                    ],
                    width = 2
                ),
                dbc.Col(
                    [   
                        html.Label('Station 2', style={'font-weight': 'bold'}),
                        dcc.Dropdown(
                            id='station_2',
                            options=station_options,
                            value=station_options[2]['value'],
                            clearable=False,
                            style={'border-color':'blue','border-width':'3px'}
                        )
                    ],
                    width = 2
                ),
            ],
            className="mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Stündliche Temperatur"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('Jahr:', style={'fontSize': '12px'}),
                                        dcc.Dropdown(
                                            id='year_dropdown_hourly',
                                            options=[{'label': i, 'value': i} for i in year_options],
                                            value=2022,
                                            clearable=False,
                                            #style={'width': '150px', 'height': '36px'}
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        html.Label('Datum:', style={'fontSize': '12px'}),
                                        dcc.DatePickerSingle(
                                            id='hourly_picker',
                                            min_date_allowed=datetime.datetime(year_options[0], 1, 1),
                                            max_date_allowed=datetime.datetime(year_options[0], 12, 31),
                                            initial_visible_month=datetime.datetime(year_options[0], 1, 1),
                                            date=datetime.datetime(year_options[0], 1, 1),
                                            display_format='DD.MM.YYYY',
                                        ),
                                    ], width=3),
                                ], align='center'),
                                dbc.Row(dcc.Graph(id='hourly_plot')),
                            ])
                        )
                    ],
                    width = 6
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Hitzewellen Identifikation"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('Jahr:', style={'fontSize': '12px'}),
                                        dcc.Dropdown(
                                            id='year_dropdown_daily',
                                            options=[{'label': i, 'value': i} for i in year_options],
                                            value=2022,
                                            clearable=False,
                                            #style={'width': '150px', 'height': '36px'}
                                        ),
                                    ], width=3),
                                    dbc.Col([
                                        html.Label('Monate:', style={'fontSize': '12px'}),
                                        dcc.RangeSlider(
                                            id='month_slider',
                                            min=1,
                                            max=12,
                                            step=1,
                                            value=[5,7],
                                            marks={i+1:elem for i, elem in enumerate(['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov','Dez'])}
                                        )
                                    ], width=9),
                                ], align='center'),
                                dbc.Row(dcc.Graph(id='daily_plot')),
                            ])
                        )
                    ],
                    width = 6
                ),
            ],
            className="mb-4"
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(markdown_referenzen),

                    ],
                    className="mt-4",                
                    )
            ],
            #className="mb-4"
        ),
        html.Div(style={'height': dis2 + 'vh'}),
    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)


####################### Callbacks #######################


@callback(
    Output('hourly_picker', 'min_date_allowed'),
    Output('hourly_picker', 'max_date_allowed'),
    Output('hourly_picker', 'initial_visible_month'),
    Output('hourly_picker', 'date'),
    Input('year_dropdown_hourly', 'value')
)
def update_date_picker(year):
    min_date = datetime.datetime(year, 1, 1)
    max_date = datetime.datetime(year, 12, 31)
    initial_month = datetime.datetime(year, 7, 1)
    date = datetime.datetime(year, 7, 15)
    return min_date, max_date, initial_month, date


####################### Hourly Plot #######################
colors = ['red', 'blue']
@callback(
    Output('hourly_plot', 'figure'),
    [Input('hourly_picker', 'date'),
     Input('station_1', 'value'),
     Input('station_2', 'value')]
)
def update_plot(selected_date, station1, station2):
    selected_date = pd.to_datetime(selected_date)
    filtered_df = hourly[hourly['TIME'].dt.date == selected_date.date()]
    
    traces = []
    for i, station_id in enumerate([station2, station1]):
        station_data = filtered_df[filtered_df['STATION_ID'] == station_id]
        trace = go.Scatter(
            x=station_data['TIME'],
            y=station_data['TEMP'],
            mode='lines',
            line=dict(color=colors[::-1][i]),
            hovertemplate=
            '<br><b>Zeit</b>: %{x}' +
            '<br><b>Temperatur</b>: %{y:.2f}°C<extra></extra>',
        )
        traces.append(trace)
    
    layout = go.Layout(
        xaxis={
            'title': 'Uhrzeit', 'gridcolor': '#F2F2F2',
            'tickvals': [selected_date + pd.Timedelta(hours=i) for i in range(0, 25, 2)],
            'ticktext': ['<br>'+str(i) for i in range(0, 25, 2)]
        },
        yaxis={'title': 'Temperatur (°C)', 'range': [-10, 40], 'gridcolor': '#F2F2F2', 'zeroline': False},
        hovermode='closest',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        
    )
    
    
    fig = go.Figure(data=traces, layout=layout)
    return fig


####################### Daily Plot #######################

@callback(
    Output('daily_plot', 'figure'),
    [Input('month_slider', 'value'),
     Input('year_dropdown_daily', 'value'),
     Input('station_1', 'value'),
     Input('station_2', 'value')]
)
def update_plot(month_range, year, station1, station2):
    filtered_df = daily[(daily['DATE'].dt.year == year) & (daily['DATE'].dt.month >= month_range[0])  & (daily['DATE'].dt.month <= month_range[1])]

    # Initialize figure with 2 subplots
    fig = make_subplots(rows=2, cols=1, shared_yaxes='rows', y_title='Tägliche Höchsttemperatur (°C)', x_title='Datum')

    for i, station_id in enumerate([station1, station2]):
        station_data = filtered_df[filtered_df['STATION_ID'] == station_id]
        trace = go.Scatter(
            x=station_data['DATE'],
            y=station_data['MAX_TEMP'],
            mode='lines',
            line=dict(color=colors[i]),
            hoverinfo=None
        )

        # Heatwave identification
        heatwave_dates = filtered_df[(filtered_df['HEATWAVE'] == 1) & (filtered_df['STATION_ID'] == station_id)]['DATE'].tolist()

        if len(heatwave_dates) > 0:
            heatwave_list = divide_dates_into_sublists(heatwave_dates)
            for h in heatwave_list:
                heatwave_trace = go.Scatter(
                    x=h,
                    y=[filtered_df.MAX_TEMP.max()] * len(h),
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='orange',
                    line=dict(color='orange'),
                    opacity=0.5,
                    hoverinfo='skip',
                    showlegend=False
                )
                #heatwave_traces.append(heatwave_trace)
                fig.add_trace(heatwave_trace, row = i+1, col=1)
        # Add trace to subplot
        fig.add_trace(trace, row=i+1, col=1)

        trace_hover = go.Scatter(
            x=station_data['DATE'],
            y=station_data['MAX_TEMP'],
            mode='lines',
            line=dict(width=0),  # Make line invisible
            marker=dict(color='rgba(0,0,0,0)'),  # Make markers invisible
            hovertemplate='Time: %{x}<br>Temperature: %{y} °C',  # Set hover text
            hoverlabel=dict(bgcolor='grey'),  # Set hover label color
            showlegend=False
        )
        # Add invisible trace to both subplots
        fig.add_trace(trace_hover, row=1, col=1)
        fig.add_trace(trace_hover, row=2, col=1)


    # Update yaxis properties
    fig.update_xaxes(tickformat="%d.%m", row=1, col=1)
    fig.update_xaxes(tickformat="%d.%m", row=2, col=1)
    fig.update_yaxes(range=[-10, 40], tickvals=[0, 15, 30])

    # Update title and color layout
    fig.update_layout(
        hovermode='closest',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.layout.annotations[0]["font"] = dict(
        family='"Open Sans", verdana, arial, sans-serif',
        size=12,
    )

    return fig