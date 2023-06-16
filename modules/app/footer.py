import dash
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc


# Create a universal footer for the app
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
