import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os

from footer import footer
from navbar import navbar

mode = "fre"

# Build the app
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.SOLAR],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# --------------------
# SIDEBAR STYLE
# --------------------

# Define the style for the sidebar and the content
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 55,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#456789",
    "zIndex": 1,  # Set the sidebar above the content
}

# Padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "zIndex": 0,  # Set the content below the sidebar
}

sidebar = sidebar = html.Div([
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("Menu", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className="bg-light",
            )], 
            style=SIDEBAR_STYLE,
)

app.layout = dbc.Container([
    dbc.Row(navbar), 
    dbc.Row([
        dbc.Col(html.Div("HeatMapper - Unveiling Munich's Hidden Heat Islands",
                         style={'fontSize':50, 'textAlign':'center'}))
    ]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2, className="sticky-top"),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    ), 
    dbc.Row(footer)
    ], 
    fluid=True
)


if __name__ == "__main__":
    if mode == "dev":
        app.run_server(debug=True, port=8050)
    else:
        app.run_server(host='0.0.0.0')


