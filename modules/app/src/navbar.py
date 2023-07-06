import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, callback
from dash_bootstrap_components._components.Container import Container
import base64
import os
import dash
from PIL import Image
from io import BytesIO


LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

logo_path = f"modules/app/src/assets/city.png"
logo = Image.open(logo_path)
buffered = BytesIO()
logo.save(buffered, format="PNG")
encoded_logo = base64.b64encode(buffered.getvalue())



# navbar = dbc.Navbar(
#     dbc.Container(
#         dbc.Row(
#             [   
#                 dbc.Col(html.P('         ')),
#                 dbc.Col(
#                     html.Img(src=LOGO, height="25px")
#                 ),
#                 dbc.Col(
#                     dbc.NavbarBrand(
#                         html.Span([
#                             "Heatmapper - ",
#                             html.I("Unveiling Munich's Hidden Heat Islands", style={'font-size': '14px', 'text-transform': 'none'})
#                         ]), className="ms-2")
#                 ),
#             ]
#         )
#     ),
#     color="primary",
#     dark=True
# )


navbar = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        # dbc.Col(
                        #     [
                        #         html.P(' ')
                        #     ],
                        #     width={"size": "auto", 'offset':'0'}
                        # ),
                        dbc.Col(
                            [
                                #html.Img(src=encoded_image, height="40px")
                                #html.Img(src=dash.get_asset_url('city.png'), height="40px")
                                html.Img(src = f"data:image/png;base64,{encoded_logo.decode()}", height="40px")
                            ],
                            width={"size": "auto", 'offset':'0'},
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        [
                                            html.Strong("Heatmapper - ", style={'font-size': '20px'}),
                                            html.I("Unveiling Munich's Hidden Heat Islands", style={'font-size': '18px', 'text-transform': 'none'})
                                        ],
                                        className="align-self-center", 
                                    )
                                ],
                                className="d-flex align-items-center"
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