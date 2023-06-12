import dash
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import folium
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
from dash_bootstrap_components import Container

# Build the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Gif url and options definition for gif
city_gif = "https://assets1.lottiefiles.com/private_files/lf30_ysi7tprv.json"
picture = "https://www.myiconstory.com/munich-onion-towers-1-png"
options_gif = dict(
    loop=True,
    autoplay=True,
    rendererSettings=dict(preserveAspectRatio="xMidYMid slice"),
)

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ms-2", n_clicks=0),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

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
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# --------------------
# FOOTER
# --------------------

footer = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # logo and name
                                # dmc.Image(width=35, height=35, src=logo),
                                dmc.Text(
                                    "Feel the heat",
                                    style={"fontSize": 20},
                                )
                            ]
                        ),
                        # columns with authors
                        dbc.Col(
                            [
                                # dmc.Text("Autoren"),
                                # Yvette
                                # DashIconify(icon="openmoji:woman-student", width=20),
                                (
                                    dmc.HoverCard(
                                        shadow="md",
                                        children=[
                                            dmc.HoverCardTarget("Stefan Grochowski"),
                                            dmc.HoverCardDropdown(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Anchor(
                                                                DashIconify(
                                                                    icon="bi:file-person",
                                                                    width=40,
                                                                ),
                                                                href="https://www.linkedin.com/in/stefan-grochowski-a13185166/",
                                                                target="_blank",
                                                            ),
                                                            dmc.Anchor(
                                                                DashIconify(
                                                                    icon="bi:github",
                                                                    width=40,
                                                                ),
                                                                href="https://github.com/stefgr1",
                                                                target="_blank",
                                                            ),
                                                        ],
                                                        p=0,
                                                    )
                                                ]
                                            ),
                                        ],
                                        position="top",
                                    )
                                ),
                            ]
                        ),
                        # column with disclaimer and uni link
                        dbc.Col(
                            [
                                dmc.Anchor(
                                    DashIconify(icon="bi:github", width=25),
                                    href="https://github.com/MGenschow/DS_Project",
                                    target="_blank",
                                ),
                                dbc.Col(
                                    [
                                        dmc.Text(
                                            [
                                                "Dieses Projekt wird im Rahmen des Studiengangs Masterstudiengangs ",
                                                dmc.Anchor(
                                                    "Data Science in Business and Economics",
                                                    href="https://uni-tuebingen.de/fakultaeten/wirtschafts-und-sozialwissenschaftliche-fakultaet/faecher/fachbereich-wirtschaftswissenschaft/wirtschaftswissenschaft/studium/studiengaenge/master/msc-data-science-in-business-and-economics/",
                                                    underline=False,
                                                ),
                                                " an der Universität Tübingen entwickelt.",
                                            ],
                                            style={"fontSize": 10},
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="fixed-bottom",
                    style={
                        "background-color": "#579852",
                        "padding": "2rem 1rem",
                        "width": "100%",
                    },
                )
            ]
        )
    ]
)


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

# Define the sidebar
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("Menu", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink(
                    "Temperature in the morning", href="/page-1", active="exact"
                ),
                dbc.NavLink(
                    "Temperature in the evening", href="/page-2", active="exact"
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Define the content layout
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# App layout
app.layout = html.Div([dcc.Location(id="url"), navbar, sidebar, content, footer])


# Define the function to render the page content
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return [
            html.Div(
                [
                    html.H3(
                        "Introduction",
                        style={"text-align": "center"},
                    ),
                    html.Div(
                        de.Lottie(
                            options=options_gif, width="30%", height="30%", url=city_gif
                        )
                    ),
                    html.P("The following tabs show a map of Munich."),
                ]
            )
        ]
    elif pathname == "/page-1":
        return [
            html.Div(
                [
                    html.H3(
                        "Temperature in the morning", style={"text-align": "center"}
                    ),
                    html.Iframe(
                        id="map",
                        srcDoc=open("morning.html", "r").read(),
                        width="100%",
                        height="600",
                        className="align-middle",
                    ),
                ]
            )
        ]

    elif pathname == "/page-2":
        return [
            html.Div(
                [
                    html.H3(
                        "Temperature in the afternoon", style={"text-align": "center"}
                    ),
                    html.Iframe(
                        id="map",
                        srcDoc=open("afterNoon.html", "r").read(),
                        width="100%",
                        height="600",
                        className="align-middle",
                    ),
                ]
            )
        ]
    return [
        html.Div(
            dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognized..."),
                ]
            )
        )
    ]


if __name__ == "__main__":
    app.run_server(debug=True, port=8081, use_reloader=False)
