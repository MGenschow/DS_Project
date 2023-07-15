import dash
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime

from dash_iconify import DashIconify
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback, callback_context
from dash_bootstrap_components import Container
from pathlib import Path
from dash.exceptions import PreventUpdate
from datetime import  timedelta
from plotly.subplots import make_subplots

from root_path import *

dash.register_page(__name__,
                   path='/Hintergrund',  # '/' is home page and it represents the url
                   name='Hintergrund',  # name of page, commonly used as name of link
                   title='Hintergrund',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Blog about Project',
                   icon="fa-solid fa-circle-info", 
                   order=5
)

md_disclaimer = """
<small><em>
Dieser Part der Website konzentriert sich auf technische und statistische Aspekte unseres Projekts.
Um hier ein größtmögliches Publikum zu erreichen ist der Blogpost in Englisch geschrieben.
<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Flag_of_the_United_Kingdom_%281-2%29.svg/1200px-Flag_of_the_United_Kingdom_%281-2%29.svg.png' width='15' height='10' />
</em></small>
"""

md_ref = '''
Gasparrini, A. and Armstrong, B. (2011). The impact of heat waves on mortality. *Epidemiology*, 22(1):68.  
Huth, R., Kyselyy, J., and Pokorna, L. (2000). A GCM simulation of heat waves, dry spells, and their relationships to circulation. *Climatic Change*, 46(1-2):29–60.  
Kysely, J. (2004). Mortality and displaced mortality during heat waves in the Czech Republic. *International Journal of Biometeorology*, 49:91–97.  
Kysely, J. (2010). Recent severe heat waves in central Europe: How to view them in a long-term prospect? *International Journal of Climatology: A Journal of the Royal Meteorological Society*, 30(1):89–109.  
Meehl, G. A. and Tebaldi, C. (2004). More intense, more frequent, and longer lasting heat waves in the 21st century. *Science*, 305(5686):994–997.
'''

md_introduction = """
As we experience both rapid urbanization and climate change globally, it is critical to understand the numerous environmental implications it brings.
A prominent phenomenon resulting from this is Urban Heat Islands (UHIs), urban areas significantly warmer than their rural surroundings due to human activities.
Among various contributing factors, land cover and land use (LCLU) characteristics of urban landscapes play a significant role in modulating UHI intensity.
Despite the recognition of this relationship, adequate feature extraction and quantifying the effects of various LCLU characteristics remain a challenging task.
In this article, we take a data science approach to this complex problem, aiming to develop a model that captures the influence of LCLU characteristics on urban heat intensity.
Our investigation involves rigorous statistical analysis and complex data manipulation, necessitating proficient understanding in both areas.

By leveraging high-quality geospatial data and state-of-the-art statistical modeling techniques, we seek to gain a deeper understanding of the intricate relationship between LCLU and UHI.
In doing so, we hope to provide a comprehensive account of urban heat phenomena and its potential mitigating factors.
It is our hope that this will not only stimulate further academic research but also inform urban planning policies for mitigating the impacts of urban heat islands.

This post will guide you through our data gathering, processing, and modeling steps. It is strutured as follows:
"""

md_hw = """
The urban heat island effect is particularly problematic when extreme temperatures occur on consecutive days (Gasparrini and Armstrong, 2011).
These periods of extreme heat are typically referred to as heatwaves.
Consequently, our entire analysis is focused on temperature data recorded during a heatwave.
For our analysis, we followed the conventional definition of heatwave given by Huth et al. (2000), which is most commonly used in the meteorological literature for Central Europe (Meehl and Tebaldi, 2004; Kysely, 2004, 2010).
This definition is as follows:
"""

md_hw_cit = """
<blockquote style='background-color:#f9f9f9; padding:10px; margin:1em;'>
A heatwave is declared when a temperature of 30°C is exceeded for at least three consecutive days. The heatwave continues as long as the average maximum temperature remains above 30°C throughout the entire period, and the maximum temperature does not fall below 25°C on any single day.
</blockquote>
"""

# <img src='https://github.com/MGenschow/DS_Project/blob/main/figures/grid_element_all.JPG?raw=true' width='70%' />
md_slx_1 = """
Having now both granular temperature data and classified land cover and land use characteristics availabe, we are now armed with data that allows us to proceed to the modeling phase.
The purpose of this phase is to establish a causal relationship between these features and the observed urban heat intensity.
As a first step, the data needs to be disaggregated to have some distinct observations to work with.
We do this by laying a grid over the city of Munich and its surrounding aggregating the data within each grid cell.
This allows us to have a large number of observations to work with, while still retaining the spatial structure of the data.
The value of our dependent variable is calculated using a weighted average of all pixels that fall within a grid cell.
The independent variables are calculated by the share of surface that is covered by a certain land cover and land use class.
"""

md_slx_2 = """
Given the geospatial nature of our data, traditional linear regression models may not provide the best solution due to their inability to account for spatial dependence.
This is where spatial econometric models come into play.
They are designed to incorporate spatial dependence, allowing us to leverage the spatial structure of our data.
Among the pantheon of spatial econometric models, we have opted for the Spatial Lag of X (SLX) model.
The SLX model is a type of spatial cross-regressive model that captures spatial effects on the independent variables by additionally introducing so-called lagged independent variables.
Formally, the SLX model can be represented as:

```math
Y = Xβ + WXγ + ε
```

Where:

- `Y` is the dependent variable (in our case, temperature)
- `X` is a matrix of our explanatory variables (our land cover and land use features)
- `β` and `γ` are parameters to be estimated
- `W` is the spatial weight matrix
- `WX` represents the spatially lagged independent variables
- `ε` is the error term

The inclusion of the term `WXγ` allows us to account for spatial spillover effects, i.e., how the land use characteristics of neighboring areas influence the temperature of the area in question.
For example, a large water body in a neighboring region might influence the temperature of our area of interest.
In practice, the lagged term consists of the the sum of the independent variables of all neighboring areas where neighboring areas refer to all grid elements that enclose the grid of interest.
The model parameter can then be estimated using ordinary least squares.

To account for important interactions between the features (building, low vegetation, water, trees and roads), we interacted them in `X`.
Additionally, we apply a log transformation to all of our independent variables to allow for decreasing marginal returns.
Lastly, we include average height within a grid as a proxy for the urban canyon effect.
Interestingly, the coefficient for this variable is both economically and statistically insignificant.
"""

layout = dbc.Container(
    [
        html.Div(style={'height': '5vh'}),
        dbc.Row(
            [
                dcc.Markdown(md_disclaimer, dangerously_allow_html=True),
                dcc.Markdown(md_introduction, style={"text-align": "justify"})
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    [
                        #html.H1('Table of Contents'),
                        html.Ul([
                            html.Li(html.A('Extracting land surface temperature data from ECOSTRESS', href='#section-1')),
                            html.Li(html.A('Heatwave detection', href='#subsection-1-1')),
                            html.Li(html.A('Extracting land cover and land use data from orthophotos', href='#section-2')),
                            html.Li(html.A('Econometrics: Modeling LST using a SLX model', href='#section-3')),
                            html.Li(html.A('References', href='#section-ref'))
                        ])
                     ], id='toc'
                ),  
            ]
        ),
        dbc.Row(
            [
                html.H2('Extracting land surface temperature data from ECOSTRESS', id='section-1'),
                html.H3('Heatwave detection', id='subsection-1-1'),
                dcc.Markdown(md_hw, style={"text-align": "justify"}),
                dcc.Markdown(md_hw_cit, style={"text-align": "justify"}, dangerously_allow_html=True),
                html.H2('Extracting land cover and land use data from orthophotos', id='section-2'),
                html.H2('Econometrics: Modeling LST using a SLX model', id='section-3'),
                dcc.Markdown(md_slx_1, style={"text-align": "justify"}),
                html.Div(
                    [
                    html.Img(src='assets/grid_element_all.JPG', alt='Grid plot', width='60%'),
                    ],
                    style={'text-align': 'center'}
                ),
                dcc.Markdown(md_slx_2, style={"text-align": "justify"}),
                html.H2('References', id='section-ref'),
                dcc.Markdown(md_ref)
            ]
        ),
        html.Div(style={'height': '15vh'})
    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)
