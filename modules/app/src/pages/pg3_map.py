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
from dash.exceptions import PreventUpdate

''' kann das hier weg? Weil alle Paket sind oben mit drin!!!
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
'''
# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/Map',  # '/' is home page and it represents the url
                   name='Map with Heat information ',  # name of page, commonly used as name of link
                   title='Map',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Final map of our project'
)

# Data Import
with open('modules/app/src/assets/final_200_a.pkl', 'rb') as f:
    gdf = pickle.load(f)
gdf_json = json.loads(gdf[['geometry', 'id', 'wLST', 'impervious',
       'building', 'low vegetation', 'water', 'trees', 'road', 'ignore']].to_json())





# ESRI Tile Layer
attribution = (
    'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors,'
    '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'
)
url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, '

# Central Map Element
map_element = dl.Map(
    dl.LayersControl(
        [dl.BaseLayer(dl.TileLayer(), name = 'OpenStreetMap', checked = True)] + 
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = esri_attribution), name = 'ESRI Satellite')] + 
        [dl.Overlay(dl.LayerGroup(
            dl.GeoJSON(data=gdf_json, id="grid", options={"style":{"color":"blue", 'weight':2, 'opacity':1, 'fillOpacity': 0}})), 
                            name="Grid", 
                            checked=True)]
    ),
    center=[48.137154, 11.576124],
    zoom=13,
    id="map_2",
    style={
        "width": "100%",
        "height": "600px",
        "margin": "auto",
        "display": "block",
        "position": "relative",
    },

)

## Grid Information
# Temperature Card
temp_card = dbc.Card(
    [
        dbc.CardHeader(html.H3("Land Surface Temperature"), style={"background": "#456789"}),
        dbc.CardBody(
            [   html.Div(html.H5(id = 'grid_id')),
                html.Div(html.H4(id = 'temp_mean')),
                html.Div(html.H4("Δ Temperature: 0", id = 'temp_delta')),
                html.Div(html.H4("Predicted Temperature: 0", id = 'temp_pred')),
            ]
        )
    ],
    style={"width": "100%"},
)

# Land Cover Card
land_cover_card = dbc.Card(
    [
        dbc.CardHeader(html.H3("Land Cover"), style={"background": "#456789"}),
        dbc.CardBody(
            [  
                dcc.Graph(id='lu_bar_chart')
            ]
        )
    ],
    style={"width": "100%", 'height':'20%'},
)

# Controls Card
controls_card = dbc.Card(
    [
        dbc.CardHeader(html.H3("What-If Analysis"), style={"background": "#456789"}),
        dbc.CardBody(
            [   
                html.Div(
    [
        dcc.Slider(
            id=f'slider-{i}',
            min=0,
            max=100,
            value=20,
            step=10,
            updatemode="drag",
        )
        for i in range(6)
    ]
)
            ]
        )
    ],
    style={"width": "100%"},
)




# Storage for the land cover information to allows multiple usage of callback output
lu_storage = html.Div(id='lu_storage', style={'display': 'none'})


# Layout
# --------------------
layout = dbc.Container(
    [dbc.Row([
        dbc.Col(map_element,  width = 8),
        dbc.Col(html.Div(
                [   
                    lu_storage,
                    dbc.Row([temp_card], justify="center"),
                    html.Br(),
                    dbc.Row([land_cover_card], justify="center"),
                    html.Br(),
                    dbc.Row([controls_card], justify="center"),
                    
                ]
            ), width = 4)
    ])], 
    fluid = True
)

# --------------------


################################ Callbacks ################################

########## Grid Information ##########
@callback(
    [Output("grid_id", "children"), Output("temp_mean", "children"), Output("lu_storage", "children")],
    [Input("grid", "click_feature")],
)
def update_grid_info(click_feature):
    # Check if something is clicked
    if click_feature is not None:
        properties = click_feature["properties"]
    
    # If nothing is clicked, return default
    else:
        properties = gdf_json['features'][0]['properties']

    # Get the grid id and the land cover information
    grid_id = properties["id"]
    impervious = np.round(properties["impervious"] * 100, 2)
    building = np.round(properties["building"] * 100, 2)
    low_vegetation = np.round(properties["low vegetation"] * 100, 2)
    water = np.round(properties["water"] * 100, 2)
    trees = np.round(properties["trees"] * 100, 2)
    road = np.round(properties["road"] * 100, 2)

    return (
        f"Grid: {grid_id}",
        f"Ø Temperature: {np.round(properties['wLST'], 2)}",
        [impervious, building, low_vegetation, water, trees, road]
    )

@callback(
    Output("text-output", "children"),
    [Input("lu_storage", "children")]
)
def update_text_output(lu_values):
    return f"Land Cover: {lu_values}"


########## Bar Chart ##########
@callback(
    Output('lu_bar_chart', 'figure'),
    [Input('lu_storage', 'children')]
)
def update_bar_chart(lu_storage):
    categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
    colors = ['#ffffff', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
    figure = go.Figure(
        data=[
            go.Bar(
                y=categories[::-1],
                x=lu_storage[::-1],   
                marker_color=colors[::-1],  # bar colors
                orientation='h'  # horizontal bars
            )
        ],
        layout=go.Layout(
            showlegend=False,
            xaxis=dict(
                title='%',
                titlefont_size=12,
                tickfont_size=12,
                range=[0, 100]
            ),
            yaxis=dict(
                titlefont_size=12,
                tickfont_size=12,
                automargin=True
            ), 
            plot_bgcolor='rgba(0, 0, 0, 0)',  # make the background color transparent
            paper_bgcolor='rgba(0, 0, 0, 0)',  # make the paper background color black
            font=dict(color='white'),  # change font color to white for visib
            margin=dict(t=1, b=1, l=1, r=1), 
            bargap = 0.1,
           # height=300
        )
    )
    return figure




####################### SLIDER #############################

# @callback(
#     [Output(f'slider-{i}', 'value') for i in range(6)],
#     [Input('lu_storage', 'children')]  # 'grid_info_output' is the fifth output from 'update_grid_info' callback
# )
# def set_initial_values(data):
#     # assuming data is a list of 6 values
#     print(data)
#     return data

@callback(
    [Output(f'slider-{i}', 'value') for i in range(6)],
    [Input(f'slider-{i}', 'value') for i in range(6)],
    [State(f'slider-{i}', 'value') for i in range(6)]
)
def update_sliders(*args):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    slider_id = ctx.triggered[0]["prop_id"].split(".")[0]
    values = list(args[:6])
    old_values = list(args[6:])

    # Check if the total sum is 100
    if sum(values) == 100:
        return values

    # Get the index of the changed slider
    changed_index = int(slider_id.split('-')[1])

    # Calculate the difference caused by the change of the slider
    diff = 100 - sum(values)

    # Distribute the difference among the remaining sliders
    for i in range(6):
        if i != changed_index:
            values[i] += diff / (6 - 1)

    # Check if any value goes below 0 or above 100 and adjust if necessary
    for i in range(6):
        if i != changed_index and (values[i] < 0 or values[i] > 100):
            extra = values[i] if values[i] < 0 else values[i] - 100
            values[i] -= extra
            values[changed_index] -= extra

    # Ensure that due to rounding the total sum is 100
    values[changed_index] -= sum(values) - 100

    return values