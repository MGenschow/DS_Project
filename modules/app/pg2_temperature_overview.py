import dash
import dash_leaflet as dl
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc



# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/temperature',  # '/' is home page and it represents the url
                   name='Comparison of temperatures during the day',  # name of page, commonly used as name of link
                   title='Map of temperatures',  # title that appears on browser's tab
                   image='pg1.png',  # image in the assets folder
                   description='Maps that display the temperature in the morning and afternoon.'
)

layout = html.Div(
    [
        html.H3(
            "Temperature comparison",
            style={"text-align": "center"},
        ),
        html.Br(),html.Br(),
        html.H5(
            "Temperature in the morning",
            style={"text-align": "center"},
        ),
        html.Iframe(
            id="morning_map",
            srcDoc=open("/Users/skyfano/Documents/Data_Science_Project/DS_Project/modules/app/src/assets/morning.html", "r").read(),
            width="100%",
            height="600",
            className="align-middle",
        ),
        html.Br(),
        html.Br(),
        html.H5(
            "Temperature in the afternoon",
            style={"text-align": "center"},
        ),
        html.Br(),
        html.Iframe(
            id="afternoon_map",
            srcDoc=open("/Users/skyfano/Documents/Data_Science_Project/DS_Project/modules/app/src/assets/afternoon.html", "r").read(),
            width="100%",
            height="600",
            className="align-middle",
        ),
    ]
)
