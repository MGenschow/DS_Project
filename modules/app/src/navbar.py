import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, callback
from dash_bootstrap_components._components.Container import Container

LOGO = "modules/app/src/assets/city.png"



navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [   
                dbc.Col(html.P('         ')),
                dbc.Col(
                    html.Img(src=LOGO, height="25px")
                ),
                dbc.Col(
                    dbc.NavbarBrand(
                        html.Span([
                            "Heatmapper - ",
                            html.I("Unveiling Munich's Hidden Heat Islands", style={'font-size': '14px', 'text-transform': 'none'})
                        ]), className="ms-2")
                ),
            ]
        )
    ),
    color="primary",
    dark=True
)


