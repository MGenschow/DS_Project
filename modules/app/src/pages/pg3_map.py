import dash
import dash_leaflet as dl
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback
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




attribution = (
    'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors,'
    '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'
)

#url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"


# ESRI Tile Layer
url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, ' \

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
    zoom=12,
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
# Create a text output that returns the name of the input image when clicked
text_output = html.Div(
    id="text-output", children="Click on the grid to get its land cover data."
)

# Create an empty image container
image_container = html.Div(id="image-container")

layout = dbc.Container(
    [dbc.Row([
        dbc.Col(map_element,  width = 8),
        dbc.Col(html.Div(
                [
                    text_output,
                    image_container,
                ]
            ), width = 4)
    ])], 
    fluid = True
)

# layout = html.Div(
#                 [
#                     html.H3(
#                         "Map of Munich showing the districts and their population density",
#                         style={"text-align": "center"},
#                     ),
#                     map_element,
#                     html.Br(),
#                     text_output,
#                     html.Br(),
#                     image_container,
#                 ]
#             )


# --------------------



@callback(
    [Output("text-output", "children"), Output("image-container", "children")],
    [Input("grid", "click_feature")],
)
def update_grid_info(click_feature):
    if click_feature is not None:
        properties = click_feature["properties"]
        grid_id = properties["id"]
        impervious = properties["impervious"]
        building = properties["building"]
        low_vegetation = properties["low vegetation"]
        water = properties["water"]
        trees = properties["trees"]
        road = properties["road"]

        
        image_path = f"modules/app/src/assets/assets/{grid_id}.tif"
        return (
            f"Clicked on grid {grid_id} \n Impervious: {impervious} \n Building: {building} \n Low Vegetation: {low_vegetation} \n Water: {water} \n Trees: {trees} \n Road: {road}",
            f"Image Path: {image_path}",
        )
    else:
        return "Click on the grid to get its land cover data.", None
