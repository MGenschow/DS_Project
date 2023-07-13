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
import geopandas as gpd
from dash.exceptions import PreventUpdate
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from shapely.geometry import Point, Polygon
import geopy
import yaml
import time
from shapely.geometry import Point


from dash_extensions.javascript import arrow_function

from root_path import *

# root_path = os.getcwd()
# print(root_path)


dash.register_page(__name__,
                   path='/Map',  # '/' is home page and it represents the url
                   name='Map',  # name of page, commonly used as name of link
                   title='Map',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Final map of our project',
                   icon = "fa-solid fa-map-location", 
                   order = 2
)

# Data Import
features_interact = ['building','low vegetation','water','trees','road']
features_no_interact = ['const','avg_height','lag_building','lag_low vegetation','lag_water','lag_trees','lag_road']
features = features_interact + features_no_interact

with open('C:/Users/stefan/Documents/10_DS_Project/DS_Project/modules/app/src/assets/gpd_250_e.pkl', 'rb') as f:
    d = pickle.load(f)
gdf = gpd.GeoDataFrame(d, geometry='geometry')

gdf_json = json.loads(gdf[['geometry', 'id', 'wLST', 'impervious', 'pred'] + features].to_json())

# Model Import
with open(root_path + '/assets/Causal_Model_250_e.pkl', 'rb') as f:
    model = pd.read_pickle(f)

def create_log_interactions(df, features_interact, features_no_interact, all=True):
    df_log_interact = df.copy()
    for feature in features_interact:
        df_log_interact[feature] = np.log(df_log_interact[feature] * 100 + 1)
    poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_interact = poly_features.fit_transform(df_log_interact[features_interact])
    X_interact = pd.DataFrame(X_interact, columns=poly_features.get_feature_names_out(features_interact))
    if all:
        for feature in features_no_interact:
            if feature not in ['const','avg_height']:
                df_log_interact[feature] = np.log(df_log_interact[feature] * 100 + 1)
    return pd.concat([X_interact, df_log_interact[features_no_interact]], axis=1)


###### Function and data for the adress feature
def check_coordinates_in_bbox(latitude, longitude, bounding_box):
    point = Point(longitude, latitude)
    bbox_polygon = Polygon([(bounding_box[0], bounding_box[1]), (bounding_box[0], bounding_box[3]),
                            (bounding_box[2], bounding_box[3]), (bounding_box[2], bounding_box[1])])
    return bbox_polygon.contains(point)

# Load config file
with open(root_path + '/../../config.yml', 'r') as file:
    config = yaml.safe_load(file)
# Store bounding box
bbox = config['bboxes']['munich']

df = pd.read_csv(root_path + '/assets/adressen_aktuell.txt', sep=',')
# Concat STRANAM and HSZ and store it is a list
df['Adress'] = df['STRANAM'] + ' ' + df['HSZ'] + ', München'

# Store the Adress column in a list
adressList = df['Adress'].tolist()
adress_options = [{'label':elem, 'value':elem} for elem in adressList]



####################### Map Element ##########################
# ESRI Tile Layer
attribution = ('Map: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>') # OSM Attribution

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles: &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')

# Central Map Element
map_element = dl.Map(
    [
    dl.LayersControl(
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = attribution + esri_attribution),checked=True, name = 'ESRI Satellite')] + 
        [dl.BaseLayer(dl.TileLayer(attribution = attribution), name = 'OpenStreetMap', )] + 
        [dl.Overlay(
            dl.LayerGroup(
                dl.GeoJSON(data=gdf_json, 
                            id="grid", 
                            options={"style":{"color":"#209e9e", 'weight':2, 'opacity':1, 'fillOpacity': 0}},
                            hoverStyle=arrow_function(dict(weight=3, color='#d62f2f', dashArray='')),
                            #zoomToBounds=True
                            )
                        ), 
            name="Grid", 
            checked=True)]
    ),
    dl.GeoJSON(
                        id="last_clicked_grid", 
                        options={"style":{"color":"red", 'weight':3, 'opacity':1, 'fillOpacity': 0.4}},
                        #zoomToBounds=True
                    )
    ],
    center=[48.137154, 11.576124],
    style={'width': '100%', 'height': '45vh', 'margin': "auto", "display": "block"},
    zoom=13)




########################## Storage Elements ####################
# Storage for the land cover information to allows multiple usage of callback output
lu_storage = html.Div(id='lu_storage', style={'display': 'none'})
grid_id_storage = html.Div(id = 'grid_id_storage', style={'display': 'none'})
controls_storage = html.Div(id = 'controls_storage', style={'display': 'none'})
initial_pred_storage = html.Div(id = 'initial_pred_storage', style={'display': 'none'})
base_temp_storage = html.Div(id = 'base_temp_storage', style={'display': 'none'})


# Image Containers: 
image_container = html.Div(id="image-container")
mask_container = html.Div(id="mask-container")



############################## Layout #############################
layout = dbc.Container(
    [
        # Placehoder for storage objetcs
        lu_storage, grid_id_storage, controls_storage, initial_pred_storage, base_temp_storage,
        html.Div(style={'height': '10vh'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5('Adress Search:'),
                    ],
                    width={'size':2, 'offset':0},
                    className='d-flex align-items-center'  # Align items in the center
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(id="adress_dropdown", placeholder='Start typing...'),

                    ],
                    width={'size':3, 'offset':0},
                )
            ],
            class_name='mb-2'
        ),
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
                                            html.Span(id='temp_delta', style={"font-weight": "bold", "font-size": "1.4rem"}), html.Br(),
                                            html.Span("Predicted Temperature: ", style={"font-size": "1rem"}),
                                            html.Span(id='temp_pred', style={"font-weight": "bold", "font-size": "1.4rem"})
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
        html.Div(style={'height': '10vh'}),
    ],
    style={"height": "100vh"},
    fluid=True
)



################################ Callbacks ################################
@callback(
        Output('grid_id_storage', 'children', allow_duplicate=True),
        Input('grid', 'click_feature'),
        prevent_initial_call='initial_duplicate'
)
def update_grid_id(click_feature):
    if click_feature is not None:
        return click_feature['properties']['id']
    else:
        return 1000001



########## Grid Information ##########
@callback(
    [Output("grid_id", "children"), Output("temp_mean", "children"), 
     Output("lu_storage", "children"), Output("image-container", "children"), 
     Output("mask-container", "children"), Output('controls_storage', 'children'), Output('initial_pred_storage', 'children'), Output('base_temp_storage', 'children')],
    [Input("grid_id_storage", "children")],
)
def update_grid_info(grid_id):
    if grid_id is not None:
        filtered_data = {'type': 'FeatureCollection', 'features': [feature for feature in gdf_json['features'] if feature['properties']['id'] == grid_id]}
        properties = filtered_data['features'][0]['properties']
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

    # Get characteristics for model inference
    initial_pred = properties['pred']
    avg_height = properties['avg_height']
    lag_building = properties['lag_building']
    lag_low_vegetation = properties['lag_low vegetation']
    lag_water = properties['lag_water']
    lag_trees = properties['lag_trees']
    lag_road = properties['lag_road']
    base_temp = properties['wLST']

    # Orthophoto Card Content
    image_path = root_path + f"/assets/orthophotos/{grid_id}.jpg"
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue())
    ortho_image = html.Div([
        html.H4('Orthophoto'),
        html.Img(src = f"data:image/png;base64,{encoded_image.decode()}", width = 300),
        #html.Hr(),
        html.Span(
            children=[
                "© Bayerische Vermessungsverwaltung – ",
                dcc.Link('www.geodaten.bayern.de',
                        href = 'http://www.geodaten.bayern.de',
                        target = '_blank')
            ],
            style = {"position": "absolute", 
                    "bottom": "0", 
                    "right": "0", 
                    "font-size": "0.6rem"})
        ],
    )

    # LULC Classification Card Content
    cmap = {'Versiegelt': "#cccccc", 'Gebäude': '#ff00ff',
        'Wiese':"#00ff00", 'Wasser':'#0000ff', 'Bäume':"#008200", 'Straße':"#ff0000"} 
    
    mask_path = root_path + f"/assets/predictions/{grid_id}.jpg"
    mask = Image.open(mask_path)
    buffered = BytesIO()
    mask.save(buffered, format="JPEG")
    encoded_mask = base64.b64encode(buffered.getvalue())
    mask_element = html.Div([
    html.H4('LULC Classification'),
    html.Img(src=f"data:image/png;base64,{encoded_mask.decode()}", width=300), 
    html.Br(),
    html.Br(),
    html.Div([
        html.Span([
            html.Div(children=[
                html.Span(className='color-dot', style={'backgroundColor': color}),
                html.Span(children=label)
            ], style={'display': 'flex', 'align-items': 'center', 'flex-grow': 1, "font-size": "0.8rem"})
            for label, color in cmap.items()
        ], style={'display': 'flex', 'justify-content': 'space-between'})
    ])
])


    return (
        f"{grid_id}",
        f"{np.round(properties['wLST'], 2)}°C",
        [impervious, building, low_vegetation, water, trees, road],
        ortho_image, 
        mask_element,
        [avg_height, lag_building, lag_low_vegetation, lag_water, lag_trees, lag_road],
        initial_pred,
        base_temp
    )

################### Coloring Last Clicked Grid Element #####################
@callback(
    Output('last_clicked_grid', 'data'),
    [Input('grid_id_storage', 'children')])
def subset_grid(value):
    sub_data = [feature for feature in gdf_json['features'] if feature['properties']['id'] == value]
    sub_data_json = {
        'type': 'FeatureCollection',
        'features': sub_data}
    return sub_data_json

##################### Model Inference Callback ##############################
@callback(
        [
            Output('temp_delta', 'children'), Output('temp_pred', 'children')
        ],
        [[Input('building_slider', 'value'), Input('low_vegetation_slider', 'value'), 
           Input('water_slider', 'value'), Input('trees_slider', 'value'), Input('road_slider', 'value')], 
         Input('controls_storage', 'children'),
         Input('initial_pred_storage', 'children'),
         Input('base_temp_storage', 'children')
         ]
)
def model_prediction(lulc_storage, controls, pred_initial, base):
    X = pd.DataFrame([[elem/100 for elem in lulc_storage] + [1] + controls], columns=features)
    X_log = create_log_interactions(X, features_interact, features_no_interact)
    pred = model.predict(X_log).item()

    delta = pred - pred_initial
    final_pred = base + delta

    if np.isclose(delta, 0, atol=0.01):
        delta = 0.0
        final_pred = base
    return (
        f"{np.round(delta,2)}°C", 
        f"{np.round(final_pred,2)}°C"
    )



# Initial Slider Values
@callback(
        [Output('impervious_slider', 'value'), Output('building_slider', 'value'), Output('low_vegetation_slider', 'value'), 
         Output('water_slider', 'value'), Output('trees_slider', 'value'), Output('road_slider', 'value')],
        [Input('lu_storage', 'children')]
)
def set_initial_values(lu_storage):
    return lu_storage


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


############ Adress Input #################
@callback(
    Output('grid_id_storage', 'children'),
    Input('adress_dropdown', 'value'),
    prevent_initial_call=True
    )
def adressToGrid(adress):
    if adress is None:
        raise PreventUpdate
    locator = geopy.geocoders.Nominatim(user_agent='myGeocoder')
    location = locator.geocode(adress)

    point = Point(*[location.longitude, location.latitude])
    grid_id = gdf[gdf.geometry.intersects(point)].id.item()
    return grid_id

@callback(
    Output("adress_dropdown", "options"),
    Input("adress_dropdown", "search_value")
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    elif len(search_value) < 3:
        options = []
    else:
        options = [o for o in adress_options if str.lower(str(o["label"])).startswith(str.lower(search_value))]
    return options



