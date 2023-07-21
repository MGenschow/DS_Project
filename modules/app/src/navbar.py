import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, callback
from dash_bootstrap_components._components.Container import Container
import base64
import os
import dash
from PIL import Image
from io import BytesIO
from root_path import *
from dash import dcc

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

logo_path = root_path + "/assets/logo_1_a.png"
logo = Image.open(logo_path)
buffered = BytesIO()
logo.save(buffered, format="PNG")
encoded_logo = base64.b64encode(buffered.getvalue())


navbar = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.A(
                                                # Using a dash_core_components Link
                                                dcc.Link(
                                                    children=html.Img(src=f"data:image/png;base64,{encoded_logo.decode()}", height="40px"),
                                                    href='/'  # this should be the route to your first page in the sidebar
                                                )),
                                    ],
                                    className="d-flex align-items-center justify-content-end"  # Align logo to the right
                                )
                            ],
                            width={"size": "auto", "offset": 0}
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        [
                                            html.Strong("HeatMapper - ", style={'font-size': '20px', 'font-weight': 'bold'}),
                                            html.Strong("Entdecke Münchens Hitzequellen!",
                                                   style={'font-size': '18px', 'text-transform': 'none'})
                                        ],
                                        className="align-self-center navbar-text"
                                    )
                                ],
                                className="d-flex align-items-center justify-content-center"  # Center-aligned text
                            ),
                            width={"size": "auto", "offset": 0}
                        )
                    ],
                    className="h-100"
                )
            ],
            className="fixed-top",
            style={
                "background-color": "#303E4F",
                "color": "#ffffff",
                "padding": "0.5rem",
                "width": "100vw",
                "height": "8vh"
            },
            fluid=True
        )
    ]
)






























'''
navbar = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Img(src=f"data:image/png;base64,{encoded_logo.decode()}", 
                                         style={'height': '40px', 
                                                'position': 'absolute',
                                                'top': '50%',
                                                'transform': 'translateY(-50%)',
                                                'padding-right': '20px'})  # Added padding-right
                            ],
                            width={"size": "auto", 'offset':'0'},
                            style={'height': '100%', 'position': 'relative'}  # Ensure that the column has a defined height
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        [
                                            html.Strong("Heatmapper - ", style={'font-size': '20px'}),
                                            html.I("Unveiling Munich's Hidden Heat Islands", 
                                                    style={'font-size': '18px', 'text-transform': 'none'})
                                        ],
                                        className="align-self-center", 
                                    )
                                ],
                                className="d-flex align-items-center justify-content-center"   # Center-aligned text
                            ),
                            width={"size": "auto", 'offset':'0'}
                        )
                    ],
                    className="h-100",
                )                
            ],
            className="fixed-top", style={"background-color": "#1a1a1a", 
                                           "color": "#ffffff", 
                                           'padding':'0.5rem',
                                           "width":"100vw",
                                           "height": "8vh"},
            fluid=True
        )
    ]
)
'''