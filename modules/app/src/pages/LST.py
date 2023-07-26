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
                   path='/Hitzeinseln',  # '/' is home page and it represents the url
                   name='Hitzeinseln',  # name of page, commonly used as name of link
                   title='Hitzeinseln',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Visualization of Land Surface Temperature',
                   icon="fa-solid fa-flag", 
                   order = 4
)

# Read local image file and convert to Data URL
with open(root_path + '/assets/Hitzeinsel.png', 'rb') as file:
    image_data = file.read()
    data_url_1 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/satelite.png', 'rb') as file:
    image_data = file.read()
    data_url_2 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

md_1 = '''
# Oberflächentemperatur
Die städtische Hitzeinsel, auch bekannt als "urbane Hitzeinsel" (UHI), bezieht sich auf ein Phänomen, 
bei dem städtische Gebiete im Vergleich zu den umliegenden ländlichen Regionen eine höhere Temperatur 
aufweisen. Diese Hitzeproblematik entsteht durch eine Kombination von städtischen Merkmalen und 
menschlichen Aktivitäten, die zu einer erhöhten Wärmeabsorption und -speicherung führen.
'''

md_2 = '''
Das Phänomen der urbanen Hitzeinsel lässt sich in allen modernen Städten beobachten, 
die sich durch ihre dicht bebauten Innenstädte auszeichnen. Die hohe Konzentration von Gebäuden, 
Straßen und Betonflächen, die Wärme absorbieren und speichern sind hiermit entscheidende Faktoren 
der Entwicklung von urbanen Hitzeinseln. Gleichzeitig begrenzt die Gebäudestruktur die Luftzirkulation 
und verhindert den natürlichen Austausch von Wärme und Feuchtigkeit. Die Verwendung von Materialien 
wie Beton und Asphalt verstärkt diesen Effekt, da sie dazu neigen, Wärme zu absorbieren und abzugeben.
'''

md_3 = '''
Um den Effekt der urbanen Hitzeinsel aufzuzeigen, benötigen wir möglichst granulare bzw. flächendeckende 
Temperaturdaten. Eine vergleichsweise hohe Granularität ist mit öffentlich Zugänglichen [Wetterdaten](/Hitzewellen) nicht 
zu erreichen. Aus diesen Gründen haben wir uns in diesem Projekt dazu entschieden, die 
Oberflächentemperatur als Proxy der tatsächlichen Temperatur zu verwenden, die mit einer Genauigkeit von
einer Temperaturmessung pro 70x70m Quadrant flächendeckende Messungen ermöglicht. 
'''

md_4 = '''
Die Landoberflächentemperatur (Land Surface Temperature, kurz: LST) ist die i.d.r. von Satelliten gemessene 
Temperatur der Erdoberfläche. Sie spielt eine entscheidende Rolle für das Verständnis und die Überwachung 
des Klimasystems der Erde, da sie wertvolle Erkenntnisse über den  Zustand der Umwelt und ihre historische 
Veränderungen liefert. Die von uns verwendeten Variablen beziehen wir vom ECOSTRESS (Ecosystem Spaceborne 
Thermal Radiometer Experiment on Space Station) Projekt der NASA. Unsere Daten werden im Rohformat bezogen und 
müssen zur adäquaten Verwendung transformiert und geographisch projiziert werden. Die Auswirkungen der 
Landoberflächentemperatur sind weitreichend und umfassen verschiedene Aspekte sowohl natürlicher als auch 
menschlicher Systeme.
'''

md_5 = '''
Die im folgenden verwendeten Daten beziehen sich auf den Sommer 2022 (1. Juni 2022 bis 31. August 2022). 
Unterschieden wird weiter zwischen Messungen, die in Hitzeperioden fallen (Link zu DWD einfügen) und Daten, 
die im Sommer liegen aber nicht in Hitzeperioden fallen (außerhalb von Hitzewellen). Diese Unterscheidung 
verdeutlicht den einschneidenden Effekt von Hitzewellen. Zusätzlich wird zwischen Messungen in den 
Morgenstunden und nachmittags unterschieden. 
'''

markdown_explanation = '''
# Oberflächentemperatur
Die Landoberflächentemperatur (Land Surface Temperature, kurz: LST) ist die i.d.R. von Sateltiten gemessene 
Temperatur der Erdoberfläche. Sie spielt eine entscheidende Rolle für das Verständnis und die Überwachung des 
Klimasystems der Erde, da sie wertvolle Erkenntnisse über den  Zustand der Umwelt und ihre Veränderungen im Hit
Laufe der Zeit liefert. Die uns von uns verwendeten Variablen beziehen wir vom ECOSTRESS (Ecosystem Spaceborne Thermal 
Radiometer Experiment on Space Station) Projekt der NASA. Die Daten werden im Rohformat bezogen und müssen zur adäquaten 
Verwendung transformiert und geographisch projiziert werden. Die Auswirkungen der Landoberflächentemperatur sind 
weitreichend und umfassen verschiedene Aspekte sowohl natürlicher als auch menschlicher Systeme.
![alt text](data_url "Title")
'''

markdown_desciption = '''
Die im folgenden verwendeten Daten beziehen sich auf den Sommer 2022 (1. Juni 2022 bis 31. August 2023). Unterschieden 
wird weiter zwischen Messungen, die in [Hitzewellen](/Hitzewellen) fallen (Link zu DWD einfügen) und Daten, die im Sommer liegen aber 
nicht in Hitzeperioden fallen (invertierte Hitzewellen). Diese Unterscheidung verdeutlicht den einschneidenden Effekt von 
Hitzewellen. Zusätzlich wird zwischen Messungen in den Morgenstunden und nachmittags unterschieden. 
'''

############################## Layout #############################
layout = dbc.Container(
    [
        html.Div(style={'height': '10vh'}),
        dbc.Row(
            [
                dcc.Markdown(md_1, style={"text-align": "justify"}, dangerously_allow_html=True)
            ],
            className="mx-5"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(md_2, style={"text-align": "justify"}, dangerously_allow_html=True)
                    ],
                    width={'size': 6, 'offset': 0},
                ),
                dbc.Col(
                    [
                        html.Img(src=data_url_1, style={'width': '100%'}),
                        html.Div('Credits to: https://community.wmo.int/en/activity-areas/urban/urban-heat-island', style={"text-align": "center", "font-size": "12px"}),  
                    ],
                    width={'size': 6, 'offset': 0},
                ),

            ],
            className="mx-5",
            align="center"
        ),
        html.Br(),
        dbc.Row(
            [
                dcc.Markdown(md_3, style={"text-align": "justify"}, dangerously_allow_html=True)
            ],
            className= "mx-5"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Img(src=data_url_2, style={'width': '100%'}),
                        html.Div('https://www.jpl.nasa.gov/missions/ecosystem-spaceborne-thermal-radiometer-experiment-on-space-station-ecostress', style={"text-align": "center", "font-size": "12px"}),
                    ],
                    width={'size':5, 'offset':0},
                    ),
                dbc.Col(
                    [
                        dcc.Markdown(md_4, style={"text-align": "justify"}, dangerously_allow_html=True)
                    ],
                    width={'size':7, 'offset':0},
                    ),
            ],
            className="mx-5", 
            align="center"
        ),
        html.Br(),
        dbc.Row(
            [
                dcc.Markdown(md_5, style={"text-align": "justify"}, dangerously_allow_html=True)
            ],
            className= "mx-5 mb-4"
        ),
        dbc.Row(
            [
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
            className= "mx-5"
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Außerhalb von Hitzewellen"),
                        html.Iframe(id='map1', width='100%', height='500px'),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3("Innerhalb von Hitzewellen"),
                        html.Iframe(id='map2', width='100%', height='500px'),
                    ],
                    width=6,
                ),
            ],
            className="mx-5",
        ),
        html.Br(),
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
