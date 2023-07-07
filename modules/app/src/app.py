import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os

from footer import footer
from navbar import navbar


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Assuming you have defined the `dash` and `page_registry` variables

sidebar = html.Div(
    [
        html.Div(
            [   
                html.Br(),
                html.Hr(),
                html.P("Menu", className="lead"),
                dbc.Nav(
                    [
                        dbc.NavLink(
                            [
                                html.I(className=page.get("icon")),
                                html.Span(page["name"], className="ms-3"),
                            ],
                            href=page["path"],
                            active="exact",
                            className="custom-navlink",
                        )
                        for page in dash.page_registry.values()
                    ],
                    vertical=True,
                    pills=True,
                    #className="nav-list-items",
                )
            ],
            className="sidebar",
        )
    ]
)



app.layout = html.Div(
    style={"overflow": "auto"},
    children=[
        navbar,
        dbc.Container(
            [   
                #dbc.Row(navbar),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [sidebar],
                            xs=3,
                            sm=3,
                            md=1,
                            lg=1,
                            xl=1,
                            xxl=1,
                            className="sticky-top",
                        ),
                        dbc.Col(
                            [dash.page_container],
                            xs=9,
                            sm=9,
                            md=11,
                            lg=11,
                            xl=11,
                            xxl=11,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
        footer
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)