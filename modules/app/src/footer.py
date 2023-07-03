import dash
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Create a footer for the app
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
                                    "HeatMapper - Unveiling Munich's Hidden Heat Islands",
                                    style={"fontSize": 15},
                                )
                            ]
                        ),
                        # columns with authors
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
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
                                            ],
                                            width={"size": "auto"},  # Set width to "auto" to allow the HoverCard to adjust its size
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.HoverCard(
                                                    shadow="md",
                                                    children=[
                                                        dmc.HoverCardTarget("Malte Genschow"),
                                                        dmc.HoverCardDropdown(
                                                            [
                                                                dmc.Group(
                                                                    [
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:file-person",
                                                                                width=40,
                                                                            ),
                                                                            href="https://www.linkedin.com/in/malte-genschow/",
                                                                            target="_blank",
                                                                        ),
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:github",
                                                                                width=40,
                                                                            ),
                                                                            href="https://github.com/MGenschow",
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
                                            ],
                                            width={"size": "auto"},  # Set width to "auto" to allow the HoverCard to adjust its size
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.HoverCard(
                                                    shadow="md",
                                                    children=[
                                                        dmc.HoverCardTarget("Aaron Lay"),
                                                        dmc.HoverCardDropdown(
                                                            [
                                                                dmc.Group(
                                                                    [
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:file-person",
                                                                                width=40,
                                                                            ),
                                                                            href="https://www.linkedin.com/in/lay-aaron/",
                                                                            target="_blank",
                                                                        ),
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:github",
                                                                                width=40,
                                                                            ),
                                                                            href="https://github.com/AaronLay",
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
                                            ],
                                            width={"size": "auto"},  # Set width to "auto" to allow the HoverCard to adjust its size
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.HoverCard(
                                                    shadow="md",
                                                    children=[
                                                        dmc.HoverCardTarget("Stefan Glaisner"),
                                                        dmc.HoverCardDropdown(
                                                            [
                                                                dmc.Group(
                                                                    [
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:file-person",
                                                                                width=40,
                                                                            ),
                                                                            href="https://www.linkedin.com/in/stefan-glaisner-0a3894152/",
                                                                            target="_blank",
                                                                        ),
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:github",
                                                                                width=40,
                                                                            ),
                                                                            href="https://github.com/stefan-1997",
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
                                            ],
                                            width={"size": "auto"},  # Set width to "auto" to allow the HoverCard to adjust its size
                                        ),
                                    ],
                                    className="d-flex justify-content-center",  # Added this class to center the HoverCards
                                )
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
                                                "Die ist ein Projekt des Studiengangs Masterstudiengangs ",
                                                html.Br(),
                                                dmc.Anchor(
                                                    "Data Science in Business and Economics",
                                                    href="https://uni-tuebingen.de/fakultaeten/wirtschafts-und-sozialwissenschaftliche-fakultaet/faecher/fachbereich-wirtschaftswissenschaft/wirtschaftswissenschaft/studium/studiengaenge/master/msc-data-science-in-business-and-economics/",
                                                    underline=False,
                                                ),
                                                " an der Universität Tübingen.",
                                            ],
                                            style={"fontSize": 10, },
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="fixed-bottom",
                    style={
                        "background-color": "#123456",
                        "padding": "2rem 1rem",
                        "width": "100%",
                    },
                )
            ]
        )
    ]
)
