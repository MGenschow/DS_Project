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
import base64


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

with open(root_path + '/assets/cadastral_data.png', 'rb') as file:
    image_data = file.read()
    img_comparison1 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/comparison_cadastral.png', 'rb') as file:
    image_data = file.read()
    img_comparison2 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()


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

md_segmentation_into = """
Drawing on existing literature pertaining to the drivers and mitigating elements of urban heat, we utilize land cover and land use (LCLU) attributes to pinpoint key 
variables contributing to the Urban Heat Island (UHI) phenomenon. To translate this problem into the realm of data science, we design data pipelines that leverage
 official data in combination with state-of-the-art deep learning algorithms. Our main objective is to address the critical question: what proportion of the surface 
 area in a given neighborhood is occupied by diverse entities such as trees, bodies of water, buildings, low vegetation, or impervious surfaces?
Each of these elements has a distinct role in influencing the urban heat, and understanding their individual and collective impact is integral to our analysis. 
This allows us to not just understand, but also visualize the intricate influence of these features on urban temperature patterns.
"""

md_segmentation_data_sources = """
#### Data Sources
In order to classify Munich's entire surface area into distinct land cover categories, we begin by using official data from the cadastral office. This data offers 
precise geospatial information about actual land usage as recorded by the cadastral office. The high geospatial resolution and readily available shape-files 
make this data an ideal starting point for our project. However, it is only comprehensive enough for some of the land cover categories we are interested in. 
In our final classifications, we leverage the official data to pinpoint bodies of water and official roads. The limitation of the official data lies in its lack 
of detailed segregation of other land cover characteristics in residential and other urban areas. While our interest lies in identifying each individual tree in 
every backyard, the official data only classifies the entire block as a residential area. (See comparison on the right) To obtain such detailed segregation of buildings, 
trees, and low vegetation, we turn to creating our own segmentation model using high-resolution orthophotos. Orthophotos are aerial images that are subsequently 
geospatially corrected and georeferenced. With their remarkable resolution of 40cm/pixel, these orthophotos enable us to locate every single tree and grassy area. 
This helps us create an extremely precise image of land cover characteristics for the whole of Munich.
"""

md_segmentation_model = """
#### Segmentation Model
In order to achieve the high precision evident in our classification model, primarily featured in this application, we employ cutting-edge deep learning techniques 
from the realm of computer vision. These deep neural networks necessitate extensive data for training, and since no labeled data for Munich was accessible to us, 
we opted for a method known as transfer learning. This technique involves utilizing a model designed for one task and repurposing it for another task. By employing 
this method, we can train a larger model with the available data and then fine-tune it to cater to a specific task for which minimal data is available. In our context, 
we developed a training pipeline that incorporates a two-stage transfer learning approach.

After evaluating various model architectures, we decided on a DeepLabV3 segmentation model with a ResNet101 backbone. The ResNet101 backbone is initialized with 
weights that have been trained on COCO.
In the first stage, we fine-tuned our model utilizing the Land-cOVEr Domain Adaptive semantic segmentation (LoveDA) dataset, comprising 5987 high spatial resolution 
(0.3 m) remote sensing images taken from Chinese cities and rural areas, along with their corresponding pixel-wise annotations. The primary goal of this stage was 
to equip our model with the ability to learn the general feature extraction layers for a semantic segmentation task from satellite imagery. Hence, both the backbone 
and the classification head were trained.
In the second stage, this pre-trained model was adapted to the actual imagery of Munich. Naturally, in order to train the model and test its accuracy on our 
dataset, we had to procure annotations for a portion of our dataset. For this, we utilized the labeling tool cvat.ai, which facilitates efficient labeling of 
image data using segmentation algorithms like Segment Anything. In total, we acquired segmentation masks for approximately 10 square kilometers of Munich which we 
used to fine-tune our model and assess its performance. In this stage, we intended to depend on the feature extraction layers learned in the first stage and only 
adjust the classification head to accurately classify pixels in our images. The fine-tuning of the model on our own dataset resulted in an increase in accuracy ranging 
between 10-40 percentage points depending on the category and an increase in average pixel-wise accuracy of 12 percentage points.
"""

layout = dbc.Container(
    [
        html.Div(style={'height': '10vh'}),
        dbc.Row(
            [
                dcc.Markdown(md_disclaimer, dangerously_allow_html=True),
                dcc.Markdown(md_introduction, style={"text-align": "justify"})
            ], 
            className="mx-5"
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
            ], 
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                html.H2('Extracting land surface temperature data from ECOSTRESS', id='section-1'),
                html.H3('Heatwave detection', id='subsection-1-1'),
                dcc.Markdown(md_hw, style={"text-align": "justify"}),
                dcc.Markdown(md_hw_cit, style={"text-align": "justify"}, dangerously_allow_html=True), 
            ], 
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                html.H2('Extracting land cover and land use data from orthophotos', id='section-2'),
                dcc.Markdown(md_segmentation_into, style={"text-align": "justify"})
            ], 
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [   
                        dcc.Markdown(md_segmentation_data_sources, style={"text-align": "justify"})
                    ],
                    width = 8
                ),
                dbc.Col(
                    [
                        dbc.Carousel(
                            items=[
                                {
                                    "key": "1",
                                    "src": img_comparison2,
                                    "header": "",
                                    "caption": "Our Segmentation",
                                    "imgClassName": "carousel-image"
                                },
                                {
                                    "key": "2",
                                    "src": img_comparison1,
                                    "header": "",
                                    "caption": "Official Cadastral Data",
                                    "imgClassName": "carousel-image"
                                },
                            ],
                            style={"max-height": "500px", "overflow": "hidden", 'width': '100%'}
                        )
                    ],
                    width = {'size':4, 'offset': 0, 'md':'auto'},
                    style = {'display': 'flex', 'align-items': 'center'}  # Center items vertically
                ),
            ], 
            className="mx-5 mb-4"
        ),



        dbc.Row(
            [
                dcc.Markdown(md_segmentation_model, style={"text-align": "justify"})
            ], 
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
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
            ], 
            className="mx-5 mb-4"
        ),
        html.Div(style={'height': '15vh'})
    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)
