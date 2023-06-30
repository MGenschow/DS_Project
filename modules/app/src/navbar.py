import dash
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Url for picture
picture = "https://www.myiconstory.com/munich-onion-towers-1-png"

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=picture, height="30px")),
                        dbc.Col(
                            dbc.NavbarBrand("Feel the heat", className="ms-0 bg-light")
                        ),
                    ],
                    align="start",
                    className="g-0",
                ),
                href="",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        ]
    ),
    color="dark",
    dark=True,
)


# add callback for toggling the collapse on small screens
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
