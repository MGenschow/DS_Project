import dash
import dash_leaflet as dl
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
from dash_bootstrap_components import Container
import base64
from PIL import Image
from io import BytesIO
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import datetime
from datetime import  timedelta
from plotly.subplots import make_subplots
from dash_extensions.javascript import arrow_function

from root_path import *

###### Data Import
with open(root_path + '/assets/station_meta.pkl', 'rb') as f:
    meta = pd.read_pickle(f)
with open(root_path + '/assets/hourly.pkl', 'rb') as f:
    hourly = pd.read_pickle(f)
with open(root_path + '/assets/daily.pkl', 'rb') as f:
    daily = pd.read_pickle(f)
daily['DATE'] = pd.to_datetime(daily['DATE'], format='%Y-%m-%d %H')

# Dropdown Options
station_options = []
for i, row in meta[['STATIONSNAME', 'STATIONS_ID']].iterrows():
    station_options.extend([{'label': row['STATIONSNAME'], 'value': row['STATIONS_ID']}])

year_options = list(range(pd.to_datetime(daily.DATE).dt.year.min(), pd.to_datetime(daily.DATE).dt.year.max()+1))

dash.register_page(__name__,
                   path='/DWD',  # '/' is home page and it represents the url
                   name='DWD',  # name of page, commonly used as name of link
                   title='DWD',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='DWD data to identify heatwaves',
                   icon="fa-solid fa-temperature-high", 
                   order=3
)

# Plotting utilities
def divide_dates_into_sublists(dates):
    """
    Divide a list of dates into sublists based on consecutive days.

    Args:
        dates (list): A list of dates in ascending order.

    Returns:
        list: A list of sublists, where each sublist contains consecutive dates.

    """
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


####################### Map Element ##########################
# ESRI Tile Layer
attribution = ('Map: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>') # OSM Attribution

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles: &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')
dwd_attibution = (' | Stations: &copy; <a href="https://opendata.dwd.de/">DWD Open Data</a>')
# Central Map Element
map_element = dl.Map(
    [
    dl.LayersControl( 
        [dl.BaseLayer(dl.TileLayer(attribution = attribution + dwd_attibution), name = 'OpenStreetMap', checked=True)] +
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = attribution + esri_attribution + dwd_attibution), name = 'ESRI Satellite')] 
    )],
    center=[48.137154, 11.576124],
    style={'width': '100%', 'height': '40vh', 'margin': "auto", "display": "block"},
    zoom=9)

for index, row in meta.iterrows():
    # Get the latitude and longitude values
    latitude = row['GEOBREITE']
    longitude = row['GEOLAENGE']
    
    # Create a marker for each location
    marker = dl.Marker(position=[latitude, longitude], children=[
        dl.Tooltip(html.Div([
            html.B('Stationsname:'), ' ' + str(row['STATIONSNAME']),
            html.Br(),
            html.B('Aktiv seit '), ' ' + str(row['VON_DATUM']),
            html.Br(),
            html.B('Höhe:'), ' ' + str(row['STATIONSHOEHE']),
        ]))
    ])
    
    # Add the marker to the map
    map_element.children.append(marker)



############################## Layout #############################
layout = dbc.Container(
    [   
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("DWD Temperatur Daten"),
                        html.P("The urban heat island effect is particularly problematic if temperatures are high on consecutive days (Gasparrini and Armstrong, 2011). Such periods are also conventionally referred to as heat waves. Therefore, our analysis of urban heat intensity is based on temperature data that was recorded during the time of a heat wave. We follow the definition by Huth et al. (2000) which has been frequently applied in metereological literature (Meehl and Tebaldi, 2004; Kysely, 2004, 2010). It goes as follows: A heat wave is detected as soon as a temperature of 30°C is exceeded for at least three consecutive days and lasts as long as the average maximum temperature remains above 30°C and does not fall below a maximum temperature of 25°C on any day. On the right you can see the three official DWD weather stations that reside in our area of interest within and around the city of Munich.",
                               style={"text-align": "justify"}),

                    ],
                    className="mt-4",                
                    )
            ],
            #className="mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("The urban heat island effect is particularly problematic if temperatures are high on consecutive days (Gasparrini and Armstrong, 2011). Such periods are also conventionally referred to as heat waves. Therefore, our analysis of urban heat intensity is based on temperature data that was recorded during the time of a heat wave. We follow the definition by Huth et al. (2000) which has been frequently applied in metereological literature (Meehl and Tebaldi, 2004; Kysely, 2004, 2010). It goes as follows: A heat wave is detected as soon as a temperature of 30°C is exceeded for at least three consecutive days and lasts as long as the average maximum temperature remains above 30°C and does not fall below a maximum temperature of 25°C on any day. On the right you can see the three official DWD weather stations that reside in our area of interest within and around the city of Munich.",
                               style={"text-align": "justify"}),
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
        html.Hr(),
        html.Br(),
    ],
    style={"height": "100vh"},
    fluid=True,
    className="mt-5"
)


############################## Callbacks #############################
# Update hourly plot date picker
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

################# Hourly Plot #################
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

################# Daily Plot #################
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
                    y=[filtered_df.MAX_TEMP.max().item()] * len(h),
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
    #fig.update_yaxes(title_text="Temperatur (°C)", range=[-10, 40],col=1)
    #fig.update_yaxes(title_text="Temperature (°C)", range=[-10, 40], row=2, col=1)
    fig.update_xaxes(tickformat="%d.%m", row=1, col=1)
    fig.update_xaxes(tickformat="%d.%m", row=2, col=1)
    fig.update_yaxes(range=[-10, 40], tickvals=[0, 15, 30])

    # Update title and color layout
    fig.update_layout(
        #tickformat="%d.%m",
        hovermode='closest',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.layout.annotations[0]["font"] = dict(
        family='"Open Sans", verdana, arial, sans-serif',
        size=14,
    )

    return fig
