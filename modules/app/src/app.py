import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from footer import footer
from navbar import navbar

# Build the app
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

sidebar = dbc.Nav(
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
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    ), 

    dbc.Row(footer)

], fluid=True)


if __name__ == "__main__":
    app.run(debug=True)