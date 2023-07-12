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

from root_path import *


dash.register_page(__name__,
                   path='/LST',  # '/' is home page and it represents the url
                   name='LST',  # name of page, commonly used as name of link
                   title='LST',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Visualization of Land Surface Temperature',
                   icon="fa-solid fa-satellite", 
                   order = 4
)


############################## Layout #############################
layout = dbc.Container(
    [
        html.Div(style={'height': '10vh'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Land Surface Temperature"),
                        html.P("Land Surface Temperature (LST) refers to the temperature of the Earth's surface as measured from space or from the ground. It plays a crucial role in understanding and monitoring the Earth's climate system, as it provides valuable insights into the state of the environment and its changes over time. The effect of Land Surface Temperature is far-reaching and encompasses various aspects of both natural and human systems."),
                        dcc.Dropdown(
                            id='time-dropdown',
                            options=[
                                {'label': 'Morning', 'value': 'morning'},
                                {'label': 'Afternoon', 'value': 'afternoon'}
                            ],
                            value='morning',
                            clearable=False,
                            style={'width': '200px'}
                        )
                    ],
                    width=12,
                    #className="mt-4"
                ),
            ],
            #className="mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Outside Heatwaves"),
                        html.Iframe(id='map1', width='100%', height='500px'),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H2("Inside Heatwaves"),
                        html.Iframe(id='map2', width='100%', height='500px'),
                    ],
                    width=6,
                ),
            ],
            #className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Temperature Hotspots in Munich"),
                        html.P("As the scorching summer sun reaches its zenith, urban areas become veritable hotspots of heat. The combination of dense concrete structures, asphalt roads, and limited green spaces creates unique microclimates that significantly impact the temperature within cities. In this article, we delve into the phenomenon of temperature hotspots in urban areas during the summer season, examining their causes and implications."),
                    ],
                    width=12
                ),
                 dbc.Col(html.Iframe(src=root_path + '/assets/avgAfterNoon_HW.html', width='100%', height='500px')),
            ],
            #className="mb-4",
        ),
         html.Div(style={'height': '10vh'}),

    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)


@callback(
    [Output('map1', 'src'), Output('map2', 'src')],
    [Input('time-dropdown', 'value')]
)
def update_maps(selected_time):
    if selected_time == 'morning':
        return (root_path + '/assets/avgMorning_nonHW.html', root_path + '/assets/avgMorning_HW.html')
    elif selected_time == 'afternoon':
        return (root_path +'/assets/avgAfterNoon_nonHW.html', root_path + '/assets/avgAfterNoon_HW.html')
    else:
        return (root_path +'/assets/avgMorning_nonHW.html', root_path + '/assets/avgMorning_HW.html') # default
