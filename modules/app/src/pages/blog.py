import dash
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime

from dash_iconify import DashIconify
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback, callback_context
from dash_bootstrap_components import Container
from pathlib import Path
from dash.exceptions import PreventUpdate
from datetime import  timedelta
from plotly.subplots import make_subplots

from root_path import *

dash.register_page(__name__,
                   path='/Blog',  # '/' is home page and it represents the url
                   name='Blog',  # name of page, commonly used as name of link
                   title='Blog',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Blog about Project',
                   icon="fa-solid fa-temperature-high", 
                   order=5
)

intro = """
# Technischer Hintergrund
Auf den folgenden Seiten wird der technische Hintergrund des Projektes erläutert. Die erklärung ist dabei in die folgenden
 Abschnitte unterteilt:
"""

motivation1 = """
# Motivation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, nisl quis tincidunt ultricies, 
nunc nisl ultricies nunc, quis aliquam nunc nisl quis nisl. Sed euismod, nisl quis tincidunt ultricies,
    nunc nisl ultricies nunc, quis aliquam nunc nisl quis nisl. Sed euismod, nisl quis tincidunt ultricies,
    nunc nisl ultricies nunc, quis aliquam nunc nisl quis nisl. Sed euismod, nisl quis tincidunt ultricies,
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, nisl quis tincidunt ultricies,
    nunc nisl ultricies nunc, quis aliquam nunc nisl quis nisl. Sed euismod, nisl quis tincidunt ultricies,
"""

layout = dbc.Container(
    [
        dbc.Row(
            [
                dcc.Markdown(intro, style={"text-align": "justify"})
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    [
                        #html.H1('Table of Contents'),
                        html.Ul([
                            html.Li(html.A('Section 1', href='#section-1')),
                            html.Li(html.A('Subsection 1.1', href='#subsection-1-1')),
                            #... continue for all sections and subsections
                        ])
                     ], id='toc'
                ),  
            ]
        ),
        dbc.Row(
            [
                dcc.Markdown(motivation1, style={"text-align": "justify"})
            ]
        )

    ],
    style={"height": "100vh"},
    fluid=True,
    className="mt-5"
)
