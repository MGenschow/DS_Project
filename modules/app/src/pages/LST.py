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
from sys import platform
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


markdown_explanation = '''
# Landoberflächentemperatur
Die Landoberflächentemperatur (Land Surface Temperature, kurz: LST) ist die i.d.R. von Sateltiten gemessene 
Temperatur der Erdoberfläche. Sie spielt eine entscheidende Rolle für das Verständnis und die Überwachung des 
Klimasystems der Erde, da sie wertvolle Erkenntnisse über den  Zustand der Umwelt und ihre Veränderungen im 
Laufe der Zeit liefert. Die uns von uns verwendeten Variablen beziehen wir vom ECOSTRESS (Ecosystem Spaceborne Thermal 
Radiometer Experiment on Space Station) Projekt der NASA. Die Daten werden im Rohformat bezogen und müssen zur adäquaten 
Verwendung transformiert und geographisch projiziert werden. Die Auswirkungen der Landoberflächentemperatur sind 
weitreichend und umfassen verschiedene Aspekte sowohl natürlicher als auch menschlicher Systeme.
'''

markdown_desciption = '''
Die im folgenden verwendeten Daten beziehen sich auf den Sommer 2022 (1. Juni 2022 bis 31. August 2023). Unterschieden 
wird weiter zwischen Messungen, die in Hitzeperioden fallen (Link zu DWD einfügen) und Daten, die im Sommer liegen aber 
nicht in Hitzeperioden fallen (invertierte Hitzewellen). Diese Unterscheidung verdeutlicht den einschneidenden Effekt von 
Hitzewellen. Zusätzlich wird zwischen Messungen in den Morgenstunden und nachmittags unterschieden. 
'''

############################## Layout #############################
layout = dbc.Container(
    [
        html.Div(style={'height': '10vh'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        #html.H1("Land Surface Temperature"),
                        #html.P("Land Surface Temperature (LST) refers to the temperature of the Earth's surface as measured from space or from the ground. It plays a crucial role in understanding and monitoring the Earth's climate system, as it provides valuable insights into the state of the environment and its changes over time. The effect of Land Surface Temperature is far-reaching and encompasses various aspects of both natural and human systems."),
                        dcc.Markdown(markdown_explanation, style={"text-align": "justify"}, dangerously_allow_html=True),
                        dcc.Markdown(markdown_desciption, style={"text-align": "justify"}, dangerously_allow_html=True),
                        dcc.Dropdown(
                            id='time-dropdown',
                            options=[
                                {'label': 'Morgens', 'value': 'morning'},
                                {'label': 'Nachmittags', 'value': 'afternoon'}
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
                        html.H2("Außerhalb von Hitzewillen"),
                        html.Iframe(id='map1', width='100%', height='500px'),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H2("Innerhalb von Hitzwellen"),
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
                        html.H1("Temperature Hotspots in München"),
                        html.P("As the scorching summer sun reaches its zenith, urban areas become veritable hotspots of heat. The combination of dense concrete structures, asphalt roads, and limited green spaces creates unique microclimates that significantly impact the temperature within cities. In this article, we delve into the phenomenon of temperature hotspots in urban areas during the summer season, examining their causes and implications."),
                    ],
                    width=12
                ),
                 dbc.Col(html.Iframe(src='assets/avgAfterNoon_HW.html', width='100%', height='500px')),
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
        if platform == "linux":
            return ('assets/avgMorning_nonHW.html', 'assets/avgMorning_HW.html')
        elif platform == "win32" or platform == 'darwin':
            return ('assets/avgMorning_nonHW.html', 'assets/avgMorning_HW.html')
    elif selected_time == 'afternoon':
        if platform == "linux":
            return ('assets/avgAfterNoon_nonHW.html', 'assets/avgAfterNoon_HW.html')
        elif platform == "win32" or platform == 'darwin':
            return ('assets/avgAfterNoon_nonHW.html', 'assets/avgAfterNoon_HW.html')
