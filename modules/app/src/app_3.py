import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os

from footer import footer
from navbar import navbar


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.SOLAR, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Assuming you have defined the `dash` and `page_registry` variables

sidebar = html.Div(
    [
        html.Div(
            [
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
    app.run_server(debug=True)