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

from dash_extensions.javascript import arrow_function



dash.register_page(__name__,
                   path='/DWD',  # '/' is home page and it represents the url
                   name='DWD',  # name of page, commonly used as name of link
                   title='DWD',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='DWD data to identify heatwaves',
                   icon="fa-sharp fa-solid fa-location-dot"
)


####################### Map Element ##########################
# ESRI Tile Layer
attribution = ('Map: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>') # OSM Attribution

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attribution = (' | Tiles: &copy; <a href="http://www.esri.com/">Esri</a>' '<a href="https://www.esri.com/en-us/legal/copyright-trademarks">')

# Central Map Element
map_element = dl.Map(
    [
    dl.LayersControl( 
        [dl.BaseLayer(dl.TileLayer(), name = 'OpenStreetMap', checked=True)] +
        [dl.BaseLayer(dl.TileLayer(url = url, attribution = attribution + esri_attribution), name = 'ESRI Satellite')] 
    )],
    center=[48.137154, 11.576124],
    style={'width': '100%', 'height': '60vh', 'margin': "auto", "display": "block"},
    zoom=13)



############################## Layout #############################
layout = dbc.Container(
    [
        # Placehoder for storage objetcs
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    map_element,
                    width={'size':12, 'offset':0},
                ),
            ],
        ),
        html.Hr(),
        html.Br(),
        html.Br(),
    ],
    style={"height": "100vh"},
    fluid=True
)

