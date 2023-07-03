import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import os
from dash_iconify import DashIconify

from footer import footer
from navbar import navbar


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.SOLAR, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Assuming you have defined the `dash` and `page_registry` variables

def get_icon(icon):
    return DashIconify(icon=icon, height=16)

sidebar = html.Div(
    [
        html.Div(
            [
                html.Hr(),
                html.P("Menu", className="lead"),
                dbc.Nav(
                    [
                        dmc.NavLink(
                                label=page["name"],
                                icon=get_icon(icon=page.get("icon")),
                   
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

'''
html.Div(
    [
        dmc.NavLink(
            label="With icon",
            icon=get_icon(icon="bi:house-door-fill"),
        ),
        dmc.NavLink(
            label="With right section",
            icon=get_icon(icon="tabler:gauge"),
            rightSection=get_icon(icon="tabler-chevron-right"),
        ),
        dmc.NavLink(
            label="Disabled",
            icon=get_icon(icon="tabler:circle-off"),
            disabled=True,
        ),
        dmc.NavLink(
            label="With description",
            description="Additional information",
            icon=dmc.Badge(
                "3", size="xs", variant="filled", color="red", w=16, h=16, p=0
            ),
        ),
        dmc.NavLink(
            label="Active subtle",
            icon=get_icon(icon="tabler:activity"),
            rightSection=get_icon(icon="tabler-chevron-right"),
            variant="subtle",
            active=True,
        ),
        dmc.NavLink(
            label="Active light",
            icon=get_icon(icon="tabler:activity"),
            rightSection=get_icon(icon="tabler-chevron-right"),
            active=True,
        ),
        dmc.NavLink(
            label="Active filled",
            icon=get_icon(icon="tabler:activity"),
            rightSection=get_icon(icon="tabler-chevron-right"),
            variant="filled",
            active=True,
        ),
    ],
    style={"width": 240},
)



'''







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