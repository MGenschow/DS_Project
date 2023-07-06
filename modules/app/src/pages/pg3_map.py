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


dash.register_page(__name__,
                   path='/Map',  # '/' is home page and it represents the url
                   name='Map',  # name of page, commonly used as name of link
                   title='Map',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Final map of our project',
                   icon = "fa-sharp fa-solid fa-map-location-dot"
)

# Data Import
with open('modules/app/src/assets/final_200_a.pkl', 'rb') as f:
    gdf = pd.read_pickle(f)
gdf_json = json.loads(gdf[['geometry', 'id', 'wLST', 'impervious',
       'building', 'low vegetation', 'water', 'trees', 'road', 'ignore']].to_json())


####################### Map Element ##########################
# ESRI Tile Layer
attribution = ('Map: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>') # OSM Attribution

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles: &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')

# Central Map Element
map_element = dl.Map(
    dl.LayersControl(
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = attribution + esri_attribution),checked=True, name = 'ESRI Satellite')] + 
        [dl.BaseLayer(dl.TileLayer(), name = 'OpenStreetMap', )] + 
        [dl.Overlay(dl.LayerGroup(
            dl.GeoJSON(data=gdf_json, id="grid", options={"style":{"color":"blue", 'weight':2, 'opacity':1, 'fillOpacity': 0}})), 
                            name="Grid", 
                            checked=True)]
    ),
    center=[48.137154, 11.576124],
    style={'width': '100%', 'height': '45vh', 'margin': "auto", "display": "block"},
    zoom=13)


########################## Storage Elements ####################
# Storage for the land cover information to allows multiple usage of callback output
lu_storage = html.Div(id='lu_storage', style={'display': 'none'})
lu_storage_initial = html.Div(id='lu_storage_initial', style={'display': 'none'})

# Image Containers: 
image_container = html.Div(id="image-container")
mask_container = html.Div(id="mask-container")



############################## Layout #############################
layout = dbc.Container(
    [
        # Placehoder for storage objetcs
        lu_storage, lu_storage_initial,
        dbc.Row(
            [
                dbc.Col(
                    map_element,
                    width={'size':12, 'offset':0},
                ),
            ],
        ),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    #dbc.CardHeader(),
                                    dbc.CardBody([
                                        html.H4('Land Surface Temperature:'),
                                        html.Br(),
                                        html.Div([
                                            html.Span("Grid ID: ", style={"font-size": "1rem"}),
                                            html.Span(id='grid_id', style={"font-weight": "bold", "font-size": "1rem"}), html.Br(),
                                            html.Span("Ø Temperature: ", style={"font-size": "1rem"}),
                                            html.Span(id='temp_mean', style={"font-weight": "bold", "font-size": "1.4rem"}), html.Br(),                                            
                                        ]),
                                        html.Hr(),
                                        html.H4('Temperature Model:'),
                                        #html.Br(),
                                        html.Div([
                                            html.Span("Δ Temperature: ", style={"font-size": "1rem"}),
                                            html.Span("0°C", id='temp_delta', style={"font-weight": "bold", "font-size": "1.4rem"}), html.Br(),
                                            html.Span("Predicted Temperature: ", style={"font-size": "1rem"}),
                                            html.Span("0°C", id='temp_pred', style={"font-weight": "bold", "font-size": "1.4rem"})
                                        ])
                                    ])
                                ],
                                class_name="mt-1 h-100"
                            )
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody([
                                            html.Div(image_container)
                                        ],
                                        class_name = "text-center text-justify")
                                    ],
                                    class_name="mt-1 h-100"
                                )
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody([
                                            html.Div(mask_container)
                                        ],
                                        class_name = "text-center text-justify")
                                    ],
                                    class_name="mt-1 h-100"
                                )
                            ]
                        )
                    ],
                class_name='m-1'
                )
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [   
                                                dbc.Row(
                                                    html.H4('What-If Analysis Controls')
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span("Impervious Surfaces", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="impervious_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                        html.Span("Building", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="building_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                        html.Span("Low Vegetation", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="low_vegetation_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span("Water", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="water_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                        html.Span("Trees", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="trees_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                        html.Span("Road", style={"font-size": "1rem"}),
                                                                        dcc.Slider(id="road_slider", min=0, max=100, value=0),
                                                                        html.Hr(),
                                                                    ]
                                                                )
                                                            ]
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    class_name="mt-1 h-100"
                                ),
                            ]
                        ),
                         dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [   
                                                html.H4('LULC Proportions:'),
                                                html.Div(id = 'lu_progress'),
                                                html.Div(id = 'lu_progress2')
                                            ]
                                        )
                                    ],
                                    class_name="mt-1 h-100"
                                ),
                            ]
                        ),
                    ],
                    class_name='m-1'
                )
            ]
        ),
        html.Hr(),
        html.Br(),
        html.Br(),
    ],
    style={"height": "100vh"},
    fluid=True
)



################################ Callbacks ################################

########## Grid Information ##########
@callback(
    [Output("grid_id", "children"), Output("temp_mean", "children"), Output("lu_storage", "children"), 
     Output("lu_storage_initial", "children"), Output("image-container", "children"), Output("mask-container", "children")],
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

    # Orthophoto Card Content
    image_path = f"modules/app/src/assets/orthophotos/{grid_id}.png"
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    ortho_image = html.Div([
        html.H4('Orthophoto'),
        html.Img(src = f"data:image/png;base64,{encoded_image.decode()}", width = 300),
        html.Span(
            children=[
                "Bayerische Vermessungsverwaltung – ",
                dcc.Link('www.geodaten.bayern.de',
                        href = 'http://www.geodaten.bayern.de',
                        target = '_blank')
            ],
            style = {"position": "absolute", 
                    "bottom": "0", 
                    "right": "0", 
                    "font-size": "0.8rem"})
        ],
    )

    # LULC Classification Card Content
    mask_path = f"modules/app/src/assets/predictions/{grid_id}.png"
    mask = Image.open(mask_path)
    buffered = BytesIO()
    mask.save(buffered, format="PNG")
    encoded_mask = base64.b64encode(buffered.getvalue())
    mask_element = html.Div([
                html.H4('LULC Classification'),
                html.Img(src = f"data:image/png;base64,{encoded_mask.decode()}", width = 300), html.Br(),
            ])


    return (
        f"{grid_id}",
        f"{np.round(properties['wLST'], 2)}°C",
        [impervious, building, low_vegetation, water, trees, road],
        [impervious, building, low_vegetation, water, trees, road],
        ortho_image, 
        mask_element
    )


# Initial Slider Values
@callback(
        [Output('impervious_slider', 'value'), Output('building_slider', 'value'), Output('low_vegetation_slider', 'value'), 
         Output('water_slider', 'value'), Output('trees_slider', 'value'), Output('road_slider', 'value')],
        [Input('lu_storage', 'children')]
)
def set_initial_values(lu_storage_initial):
    return lu_storage_initial


#### LULC Proportions Graphs
@callback(Output('lu_progress', 'children',  allow_duplicate=True), 
          [Input('impervious_slider', 'value'), Input('building_slider', 'value'), Input('low_vegetation_slider', 'value'), 
           Input('water_slider', 'value'), Input('trees_slider', 'value'), Input('road_slider', 'value')],  prevent_initial_call=True)
def update_progress_graph(impervious_slider, building_slider, low_vegetation_slider, water_slider, trees_slider, road_slider):
    categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
    colors = ['#cccccc', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
    values = [impervious_slider, building_slider, low_vegetation_slider, water_slider, trees_slider, road_slider]
    values = [np.round(values[i], 2) for i in range(6)]
    fig = dmc.Progress(
    size=20, radius=0, 
    sections=[{'value':values[i], 'color':colors[i], 'label':categories[i], 'tooltip':f"{categories[i]}: {values[i]}%"} for i in range(6)])
    return fig

@callback(Output('lu_progress2', 'children',  allow_duplicate=True), 
          [Input('impervious_slider', 'value'), Input('building_slider', 'value'), Input('low_vegetation_slider', 'value'), 
           Input('water_slider', 'value'), Input('trees_slider', 'value'), Input('road_slider', 'value')],  prevent_initial_call=True)
def update_progress_graph(impervious_slider, building_slider, low_vegetation_slider, water_slider, trees_slider, road_slider):
    categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
    colors = ['#cccccc', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
    values = [impervious_slider, building_slider, low_vegetation_slider, water_slider, trees_slider, road_slider]
    values = [np.round(values[i], 2) for i in range(6)]
    fig = dmc.Progress(
    size=20, radius=0, 
    sections=[{'value':values[i], 'color':colors[i], 'label':values[i], 'tooltip':f"{categories[i]}: {values[i]}%"} for i in range(6)])
    return fig


################## SLIDER UPDATE TO 100 #######################
@callback(
    [Output('impervious_slider', 'value', allow_duplicate=True),
    Output('building_slider', 'value', allow_duplicate=True),
    Output('low_vegetation_slider', 'value', allow_duplicate=True),
    Output('water_slider', 'value', allow_duplicate=True),
    Output('trees_slider', 'value', allow_duplicate=True),
    Output('road_slider', 'value', allow_duplicate=True)],
    [Input('impervious_slider', 'value'),
    Input('building_slider', 'value'),
    Input('low_vegetation_slider', 'value'),
    Input('water_slider', 'value'),
    Input('trees_slider', 'value'),
    Input('road_slider', 'value')],
    prevent_initial_call=True
)
def adjust_values(*args):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    slider_id = ctx.triggered[0]["prop_id"].split(".")[0]
    values = list(args[:6])

    # Check if the total sum is 100
    if sum(values) == 100:
        return values

    # Get the index of the changed slider
    slider_list = ['impervious_slider', 'building_slider', 'low_vegetation_slider', 'water_slider', 'trees_slider', 'road_slider']
    changed_index = slider_list.index(slider_id)

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

