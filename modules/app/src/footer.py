import dash
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc


footer = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [   
                        dbc.Col(html.P('Authors:'), width={"size": "auto"}),
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
                                                                        icon="bi:linkedin",
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
                                                dmc.HoverCardTarget("Stefan Glaisner"),
                                                dmc.HoverCardDropdown(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.Anchor(
                                                                    DashIconify(
                                                                        icon="bi:linkedin",
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
                                                                        icon="bi:linkedin",
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
                                                        dmc.HoverCardTarget("Aaron Lay"),
                                                        dmc.HoverCardDropdown(
                                                            [
                                                                dmc.Group(
                                                                    [
                                                                        dmc.Anchor(
                                                                            DashIconify(
                                                                                icon="bi:linkedin",
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
                                                dmc.HoverCardTarget("Github"),
                                                dmc.HoverCardDropdown(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.Anchor(
                                                                    DashIconify(
                                                                        icon="bi:github",
                                                                        width=40,
                                                                    ),
                                                                    href="https://github.com/MGenschow/DS_Project",
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
                                    width={"size": "auto", 'offset':'1'},  # Set width to "auto" to allow the HoverCard to adjust its size
                                ),
                                dbc.Col(
                                    [
                                        dmc.Text(
                                            [
                                                "Die ist ein Projekt des Masterstudiengangs ",
                                                html.Br(),
                                                dmc.Anchor(
                                                    "Data Science in Business and Economics",
                                                    href="https://uni-tuebingen.de/fakultaeten/wirtschafts-und-sozialwissenschaftliche-fakultaet/faecher/fachbereich-wirtschaftswissenschaft/wirtschaftswissenschaft/studium/studiengaenge/master/msc-data-science-in-business-and-economics/",
                                                    underline=False,
                                                    style={'transition': 'color 0.2s'},
                                                    className='link-hover',
                                                ),
                                                " an der Universität Tübingen.",
                                            ],
                                            style={"fontSize": 10, 'text_align':"center"},
                                        )
                                    ],
                                    width={"size": "auto", 'offset':'0'}
                                ),
                    ],
                    className="d-flex justify-content-center",
                    justify="center",
                )
                
            ],
            className="fixed-bottom", style={"background-color": "#1a1a1a",  
                                             "color": "#ffffff", 
                                             'padding':'0.5rem',
                                             "width":"100vw",
                                             "height": "5vh"},
                                             
            fluid=True
        )
    ]
)
