import dash
import dash_leaflet as dl
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
from dash_bootstrap_components import Container

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='General Information',  # name of page, commonly used as name of link
                   title='Beat the heat',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Histograms are the new bar charts.',
                   icon="bi:house-door-fill"
                   #icon="fa-sharp fa-solid fa-map-location-dot"
)

# Gif url and options definition for gif
city_gif = "https://assets1.lottiefiles.com/private_files/lf30_ysi7tprv.json"
options_gif = dict(
    loop=True,
    autoplay=True,
    rendererSettings=dict(preserveAspectRatio="xMidYMid slice"),
)

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''



# Defining the layout for the information tab
layout = html.Div(
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
                    html.Br(),
                    dcc.Markdown(children=markdown_text)
                ]
            )