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
                   #icon="bi:house-door-fill"
                   icon="fa-sharp fa-solid fa-circle-info"
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

## Background

Increasing urbanisation and climate change have led to the evolving and intensification of urban 
heat islands which can be defined as local areas where the temperature is considerably higher than 
in their vicinity. High temperatures in general are inevitably associated with partly drastical 
consequences for human health. However, heat reduction measures are available that can deal with the
urban heat island effect: Increasing vegetation, cutting the amount of impervious surfaces, etc.. 
The goal of this project is to identify heat islands by analysing applicable data for the city of 
Munich and to model the impact of additional heat reduction measures on potential temperature occurences.

## Approach

This project uses land surface temperature data from Ecostress and official property data as well as orthophotos
from the Bavarian State Office for Digitisation, Broadband and Surveying. The former data source denotes the
dependent variable in our analysis. The latter two data sources were used to extract land cover / land usage (LCLU)
characteristics forming the basis of our feature set. We used pre-trained and fine-tuned neural networks to reach a
granular segregation of land cover to also detect patterns that are not stored in official data.

'''



# Defining the layout for the information tab
layout = html.Div(
    style={"overflow": "auto"},
        children=[
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
                ],
            )