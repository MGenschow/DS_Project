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
import dash_player as player


from root_path import *

dis = str(10)

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Home',  # name of page, commonly used as name of link
                   title='HeatMapper',  # title that appears on browser's tab
                   description='Home page of HeatMapper',  # description of the page
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

# YouTube video URL
video_url = "https://youtu.be/hZQ38zNvRsY"


introduction = '''
Der HeatMapper ist ein Tool, das den Zusammenhang zwischen Temperatur und Oberflächeneigenschaften analysiert. 
Hier können Nutzer wahlweise ein 250mx250m Raster auf einer Karte markieren oder eine konkrete Adresse im dafür vorgesehenen Suchfeld eingeben. 
Es kann dabei zwischen einer Satelliten- und einer klassischen Kartenansicht gewählt werden. Die Oberflächentemperatur für das gesamte Stadtgebiet von München wird bereits beim Start angezeigt.
Ist eine Adresse oder ein Raster markiert, wird das zugeordnete Raster erfasst und die gemittelte Temperatur für das ausgewählte Gebiet berechnet.

Neben dem Satellitenbild des ausgewählten Rasters liefert der HeatMapper auch eine Darstellung der klassifizierten Oberflächenelemente, deren Verteilung im Detail in einem zusätzlichen Diagramm dargestellt wird.
Mittels verschiedener Regler wird dem Nutzer ermöglicht, die Verteilung der einzelnen Elemente zu modifizieren, wobei Temperaturänderungen, von unserem Modell berechnet, sofort sichtbar werden.
'''


hitzewellen = '''
Hitzeinseln und Hitzewellen sind zwei Phänomene, die mit steigenden Temperaturen in Zusammenhang stehen.

Eine Hitzewelle bezieht sich auf eine anhaltende Periode extrem hoher Temperaturen, die im Sommer auftreten kann.
 Diese extremen Wetterbedingungen können städtische und ländliche Gebiete beeinflussen und haben Auswirkungen auf die menschliche Gesundheit, Landwirtschaft und Tierwelt.

Anders als Hitzewellen bezieht sich der Begriff "Hitzeinsel" auf einen städtischen oder bebauten Bereich, der deutlich wärmer ist als seine Umgebung. 
Dieses Phänomen wird durch eine Vielzahl von Faktoren verursacht, einschließlich der Absorption von Sonnenlicht durch Gebäude und Straßen, geringem Pflanzenwuchs und der Verdichtung
 von Gebäuden und Menschen in städtischen Gebieten.
 Hitzeinseln haben insbesondere einenn negativen Einfluss auf die menschliche Gesundheit, da sie die städtischen Bereiche nachts nicht abkühlen lassen, was zu Hitzestress und damit 
 verbundenen Gesundheitsproblemen führt.

Unsere App ermöglicht es dem Nutzer, die Auswirkungen von Hitzeinseln und Hitzewellen auf die Temperatur in München zu untersuchen und zu verstehen.
'''


markdown_referenzen = '''
Alle Bilder lizenzfrei von https://pixabay.com.
'''


card_1 =  dbc.Card(
    [
        dbc.CardImg(
            src="/assets/frauenkirche.png",
            top=True,
            style={"opacity": 0.4},
        ),
        dbc.CardImgOverlay(
            dbc.CardBody(
                [
                    html.H4("Du kannst es nicht abwarten?", className="card-title"),
                    html.P(
                        "Dann klicke hier und du gelangst direkt zum HeatMapper.",
                        className="card-text",
                        style={"text-align": "left"}, # Make sure text is left-aligned
                    ),
            dbc.Button("Teste den HeatMapper", 
                        className="me-1 custom-btn", 
                        href="/HeatMapper", active=True)
                ],
                style={
                    'position': 'absolute', 
                    'left': '0',
                    'bottom': '0',
                    'color': 'black',
                    'font-weight': 'bold',
                }, 
            )
        ),
    ],
    # choose perfect width for 13 inches screen
    style={
                "width": "100%",  # Set the width of the card to 100%
            }
)


card_2 = dbc.Card(
    [
        dbc.CardImg(
            src="/assets/olympiapark.png",
            top=True,
            style={"opacity": 0.6},
        ),
        dbc.CardImgOverlay(
            dbc.CardBody(
                [
                    html.H4("Hitzewellen und -inseln", className="card-title"),
                    html.P(
                        "Werfe einen Blick auf die Daten zu den Hitzewellen oder -inseln in München.",
                        className="card-text",
                        style={"text-align": "right"},
                    ),
                    dbc.Button("Hitzewellen", href="/Hitzewellen", className="me-2 custom-btn", active=True),
                    dbc.Button("Hitzeinseln", href="/Hitzeinseln", className="me-1 custom-btn", active=True),
                ],
                style={
                    'text-align': 'right', # Make sure text is right-aligned
                    'position': 'absolute', 
                    'right': '0',
                    'bottom': '0',
                    'color': 'white',
                },
            ),
        ),
    ],
    style={"width":"100%"},
)


# Define new layout 
layout = dbc.Container(
    [
        html.Div(style={'height': dis + 'vh'}),
        html.H1("Willkommen bei HeatMapper!", className="display-5 mx-5"),
        dbc.Row(
            [
                dbc.Col(
                    [   html.H2("Was ist der Heatmapper?", className = "mb-4"),
                        card_1
                    ],
                    className="mt-4",
                    width=6,  # Set the width of the column to take 6 out of 12 columns (50% width)
                ),
                dbc.Col(
                    [
                        html.Br(),
                        dcc.Markdown(introduction, className="text-sm text-md text-lg text-xl", style={"text-align": "justify"}),
                        #html.Div(de.Lottie(options=options_gif, width="10%", height="10%", url=city_gif)),
                    ],
                    className="mt-4",
                    width=6,  # Set the width of the column to take 6 out of 12 columns (50% width)
                ),

            ],
            className="mx-5",
            align="center",  # Center the columns horizontally within the row
        ),
        dbc.Row(
            [
                dbc.Col(
                    [   html.H2("Hitzewellen und -inseln", className = "mt-4 mb-4"),
                        dcc.Markdown(hitzewellen, className="text-sm text-md text-lg text-xl", style={"text-align": "justify"}),
                    ],
                    className="mt-4",
                    width=6,  # Set the width of the column to take 6 out of 12 columns (50% width)
                ),
                dbc.Col(
                    [
                        card_2
                    ],
                    className="mt-4 mb-4",
                    width=6,  # Set the width of the column to take 6 out of 12 columns (50% width)
                ),
            ],
            className="mx-5",
            align="center",  # Center the columns horizontally within the row
        ),
        dbc.Row(
            [
                dbc.Col([html.Div([
                        html.H1("Schritt-für-Schritt-Anleitung", className="display-9 mb-4"),
                         player.DashPlayer(
                        id='youtube-player',
                        url=video_url,
                        controls=True,
                        width='100%',
                        height='600px'
                        )
                        ])], className = "mt-4 mb-4 mx-5")
            ]),
        dbc.Row(dcc.Markdown(markdown_referenzen, style={"text-align": "justify"}), className="mx-5"),
        html.Div(style={'height': dis + 'vh'}),

    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)

if __name__ == "__main__":
    app.run_server(debug=True)
