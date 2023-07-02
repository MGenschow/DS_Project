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
attribution = ('Map &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>') # OSM Attribution

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')

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
    zoom=13,
    id="map_2",
    style={
        "width": "100%",
        "height": "800px",
        "margin": "auto",
        "display": "block",
        "position": "relative",
    },

)

## Grid Information
# Temperature Card
temp_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Land Surface Temperature"), style={"background": "#456789"}),
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
        dbc.CardHeader(html.H4("Land Cover"), style={"background": "#456789"}),
        dbc.CardBody(
            [  
               html.Div(id = 'lu_progress'),
               html.Div(id = 'lu_progress2')
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
                    dbc.Label("Impervious", html_for="range-slider"),
                    dcc.Slider(id="impervious_slider", min=0, max=100, value=0),
                    dbc.Label("Building", html_for="range-slider"),
                    dcc.Slider(id="building_slider", min=0, max=100, value=0),
                    dbc.Label("Low Vegetation", html_for="range-slider"),
                    dcc.Slider(id="low_vegetation_slider", min=0, max=100, value=0),
                    dbc.Label("Water", html_for="range-slider"),
                    dcc.Slider(id="water_slider", min=0, max=100, value=0),
                    dbc.Label("Trees", html_for="range-slider"),
                    dcc.Slider(id="trees_slider", min=0, max=100, value=0),
                    dbc.Label("Road", html_for="range-slider"),
                    dcc.Slider(id="road_slider", min=0, max=100, value=0),
                ],
                className="mb-3",
                ),
                html.Div(id='hidden_div', style={'display':'none'}),
            ]
        )
    ],
    style={"width": "100%"},
)




# Storage for the land cover information to allows multiple usage of callback output
lu_storage = html.Div(id='lu_storage', style={'display': 'none'})
lu_storage_initial = html.Div(id='lu_storage_initial', style={'display': 'none'})


# Layout
# --------------------
layout = dbc.Container(
    [dbc.Row([
        dbc.Col(map_element,  width = 8),
        dbc.Col(html.Div(
                [   
                    lu_storage, lu_storage_initial,
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
    [Output("grid_id", "children"), Output("temp_mean", "children"), Output("lu_storage", "children"), Output("lu_storage_initial", "children")],
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
        [impervious, building, low_vegetation, water, trees, road],
        [impervious, building, low_vegetation, water, trees, road],

    )


########## Progress Graph ##########
# @callback(
#         Output('lu_progress', 'children'),
#         [Input('lu_storage', 'children')]
# )
# def update_progress_graph(lu_storage):
#     categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
#     colors = ['#cccccc', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
#     lu_storage = [np.round(lu_storage[i], 2) for i in range(6)]
#     fig = dmc.Progress(
#     size=20, radius=0,
#     sections=[{'value':lu_storage[i], 'color':colors[i], 'label':categories[i], 'tooltip':f"{categories[i]}: {lu_storage[i]}%"} for i in range(6)])
#     return fig
# @callback(
#         Output('lu_progress2', 'children'),
#         [Input('lu_storage', 'children')]
# )
# def update_progress_graph(lu_storage):
#     categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
#     colors = ['#cccccc', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
#     lu_storage = [np.round(lu_storage[i], 2) for i in range(6)]
#     fig = dmc.Progress(
#     size=20, radius=0, 
#     sections=[{'value':lu_storage[i], 'color':colors[i], 'label':lu_storage[i], 'tooltip':f"{categories[i]}: {lu_storage[i]}%"} for i in range(6)])
#     return fig

########## Bar Chart ##########
# @callback(
#     Output('lu_bar_chart', 'figure'),
#     [Input('lu_storage', 'children')]
# )
# def update_bar_chart(lu_storage):
#     categories = ['Impervious', 'Building', 'Low Vegetation', 'Water', 'Trees', 'Road']
#     colors = ['#ffffff', '#ff00ff', '#00ff00', '#0000ff', '#008200', '#ff0000']  # specify your colors here
#     figure = go.Figure(
#         data=[
#             go.Bar(
#                 y=categories[::-1],
#                 x=lu_storage[::-1],   
#                 marker_color=colors[::-1],  # bar colors
#                 orientation='h'  # horizontal bars
#             )
#         ],
#         layout=go.Layout(
#             showlegend=False,
#             xaxis=dict(
#                 title='%',
#                 titlefont_size=12,
#                 tickfont_size=12,
#                 range=[0, 100]
#             ),
#             yaxis=dict(
#                 titlefont_size=12,
#                 tickfont_size=12,
#                 automargin=True
#             ), 
#             plot_bgcolor='rgba(0, 0, 0, 0)',  # make the background color transparent
#             paper_bgcolor='rgba(0, 0, 0, 0)',  # make the paper background color black
#             font=dict(color='white'),  # change font color to white for visib
#             margin=dict(t=1, b=1, l=1, r=1), 
#             bargap = 0.1,
#            # height=300
#         )
#     )
#     return figure




####################### SLIDER #############################

# Initial Slider Values
@callback(
        [Output('impervious_slider', 'value'), Output('building_slider', 'value'), Output('low_vegetation_slider', 'value'), 
         Output('water_slider', 'value'), Output('trees_slider', 'value'), Output('road_slider', 'value')],
        [Input('lu_storage', 'children')]
)
def set_initial_values(lu_storage_initial):
    return lu_storage_initial

# Slider Values
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
    #old_values = list(args[6:])

    # Check if the total sum is 100
    if sum(values) == 100:
        return values

    # Get the index of the changed slider
    slider_list = ['impervious_slider', 'building_slider', 'low_vegetation_slider', 'water_slider', 'trees_slider', 'road_slider']
    changed_index = slider_list.index(slider_id)
    #changed_index = int(slider_id.split('-')[1])

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
