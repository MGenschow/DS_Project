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

from root_path import *

dis = str(10)

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Home',  # name of page, commonly used as name of link
                   title='HeatMapper',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Home page of HeatMapper',  # description of the page
                   #icon="bi:house-door-fill"
                   icon="fa-solid fa-house", 
                   order= 1
)

# Gif url and options definition for gif
city_gif = "https://assets1.lottiefiles.com/private_files/lf30_ysi7tprv.json"
options_gif = dict(
    loop=True,
    autoplay=True,
    rendererSettings=dict(preserveAspectRatio="xMidYMid slice"),
)

markdown_background = '''
## Projekthintergrund & Zielgruppe

Klimawandel und zunehmende Urbanisierung sorgen für ein extremes Aufheizen innerstädtischer Gebiete im Vergleich zu ländlicheren und 'grüneren' Flächen.
Insbesondere im Sommer stellt dies nicht nur eine Einschränkung des täglichen Lebens sondern auch eine elementare Bedrohung der eigenen Gesundheit dar (Anderson und Bell, 2009; Basu und Samet, 2002; Basu, 2009).  
Dieses Projekt soll zum öffentlichen und wissenschaftlichen Diskurs beitragen, indem es einen direkten Zusammenhang zwischen Temperatur und Oberflächenbeschaffenheiten (hauptsächlich abgeleitet aus Landbedeckungs- und Landnutzungsmerkmale) modelliert und visualisiert.
Dabei wird der Fokus auf die Stadt München gelegt, da diese bereits über eine Vielzahl an Daten verfügt, die für die Modellierung verwendet werden können.
Insbesondere wollen wir mit einfachen Anpassungsfeatures (z.B. Erhöhung der Vegetation) die Auswirkungen auf die Temperatur für ein ausgewähltes Gebiet darstellen.
Die App soll besonders Münchner*innen helfen ein Verständnis für die zugrundeliegenden Effekte und die direkten Auswirkungen von zunehmender Oberflächenversiegelung zu entwickeln. Hinsichtlich der spärlichen wissenschaftlichen Literatur in diesem Gebiet kann das Projekte auch als Grundlage für weitere Forschung dienen.
'''

markdown_approach = '''
## Herangehensweise

Wir haben für dieses Projekt Daten zur Landoberflächentemperatur (land surface temperature) von Ecostress und amtliche Liegenschaftsdaten sowie Orthofotos des Bayerischen Landesamtes für Digitalisierung, Breitband und Vermessung verwendet.
Die erstgenannte Datenquelle stellt die abhängige Variable in unserer Analyse dar.
Die beiden letztgenannten Datenquellen wurden zur Extraktion von Merkmalen der Landbedeckung/Landnutzung (LCLU) verwendet, um unsere Einflussfaktoren zu extrahieren, die einen Effekt auf die Temperatur haben sollten.
Wir nutzen neuronale Netze, um auch Muster zu erkennen, die in offiziellen Daten nicht enthalten sind (z.B. einen Baum).
'''

markdown_referenzen = '''
## Referenzen
Anderson, B. G. and Bell, M. L. (2009). Weather-related mortality: How heat, cold, and heat waves affect mortality in the United States. *Epidemiology*, 20(2):205.  
Basu, R. (2009). High ambient temperature and mortality: A review of epidemiologic studies from 2001 to 2008. *Environmental health*, 8:1–13.  
Basu, R. and Samet, J. M. (2002). Relation between elevated ambient temperature and mortality: A review of the epidemiologic evidence. *Epidemiologic reviews*, 24(2):190–202.
'''


layout = dbc.Container(
    [
        html.Div(style={'height': dis + 'vh'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            de.Lottie(options=options_gif, width="10%", height="10%", url=city_gif)
                        ),
                        dcc.Markdown(markdown_background, style={"text-align": "justify"}),
                        dcc.Markdown(markdown_approach, style={"text-align": "justify"}),
                        dcc.Markdown(markdown_referenzen, style={"text-align": "justify"})
                    ],
                    className="mt-4",                
                    )
            ],
        ),
        html.Div(style={'height': dis + 'vh'}),
    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)
