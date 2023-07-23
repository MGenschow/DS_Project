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


with open(root_path + '/assets/Swath_picture.png', 'rb') as file:
    image_data = file.read()
    img_swath = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/geo.png', 'rb') as file:
    image_data = file.read()
    img_geo = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/lst.png', 'rb') as file:
    image_data = file.read()
    img_lst = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/cloud.png', 'rb') as file:
    image_data = file.read()
    img_cloud = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/LST_1.png', 'rb') as file:
    image_data = file.read()
    img_QC_LST_1 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/Cloud_1.png', 'rb') as file:
    image_data = file.read()
    img_QC_CLD_1 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/LST_2.png', 'rb') as file:
    image_data = file.read()
    img_QC_LST_2 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/Cloud_2.png', 'rb') as file:
    image_data = file.read()
    img_QC_CLD_2 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()


with open(root_path + '/assets/cadastral_data.png', 'rb') as file:
    image_data = file.read()
    img_comparison1 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

with open(root_path + '/assets/comparison_cadastral.png', 'rb') as file:
    image_data = file.read()
    img_comparison2 = 'data:image/png;base64,' + base64.b64encode(image_data).decode()

# Custom styles for the caption and arrow icons
caption_style = {
    "color": "red",  # Change the caption text color to red
    "font-weight": "bold",  # Optionally, you can make the text bold
}

arrow_style = {
    "filter": "invert(100%)",  # Invert the color of the arrow icons (e.g., from black to white)
    "background-color": "blue",  # Optionally, you can change the background color of the arrows on hover
}


md_disclaimer = """
<small><em>
Dieser Part der Website konzentriert sich auf technische und statistische Aspekte unseres Projekts.
Um hier ein größtmögliches Publikum zu erreichen ist der Blogpost in Englisch geschrieben.
<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Flag_of_the_United_Kingdom_%281-2%29.svg/1200px-Flag_of_the_United_Kingdom_%281-2%29.svg.png' width='15' height='10' />
</em></small>
"""

md_ref = '''
Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587. 

Gasparrini, A. and Armstrong, B. (2011). The impact of heat waves on mortality. *Epidemiology*, 22(1):68.  
Huth, R., Kyselyy, J., and Pokorna, L. (2000). A GCM simulation of heat waves, dry spells, and their relationships to circulation. *Climatic Change*, 46(1-2):29–60.  
Kikon, N., Singh, P., Singh, S. K., and Vyas, A. (2016). Assessment of urban heat islands (UHI) of Noida City, India using multi-temporal satellite data. *Sustainable Cities and Society*, 22:19–28.  
Kysely, J. (2004). Mortality and displaced mortality during heat waves in the Czech Republic. *International Journal of Biometeorology*, 49:91–97.  
Kysely, J. (2010). Recent severe heat waves in central Europe: How to view them in a long-term prospect? *International Journal of Climatology: A Journal of the Royal Meteorological Society*, 30(1):89–109.  
Meehl, G. A. and Tebaldi, C. (2004). More intense, more frequent, and longer lasting heat waves in the 21st century. *Science*, 305(5686):994–997.

Wang, J., Zheng, Z., Ma, A., Lu, X., & Zhong, Y. (2021). LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. Zenodo. https://doi.org/10.5281/zenodo.5706578
'''

md_introduction = """
As we experience both rapid urbanization and climate change globally, it is critical to understand the numerous environmental implications this brings to urban communities.
A prominent phenomenon resulting from this are Urban Heat Islands. These can be defined as urban areas that are significantly warmer than their rural surroundings due to human activities.
Among various contributing factors, land use and land cover (LULC) characteristics of urban landscapes play a significant role in modulating urban heat intensity (UHI).
Despite the recognition of this relationship, adequate feature extraction and quantifying the effects of various LULC characteristics remains a challenging task.
In this article, we take a data science approach to this complex problem, aiming to develop a model that captures the influence of LCLU characteristics on urban heat intensity for the city of Munich.
Our investigation involves rigorous statistical analysis and complex data manipulation, necessitating proficient understanding in both areas.

By leveraging high-quality geospatial data and state-of-the-art statistical modeling techniques, we seek to gain a deeper understanding of the intricate relationship between LULC and UHI.
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

md_ecostress_intro = """
To measure the impact of heatwaves on urban heat intensity with greater precision, we undertook an extensive search for data sources with a high 
spatial resolution that could effectively cover the entirety of Munich, Germany. However, as is common in many urban areas, obtaining temperature data 
from conventional air temperature measuring stations at such a fine scale proved challenging due to limited coverage and spatial granularity.
To overcome this limitation, we turned to an alternative: Land surface temperature data (LST), which serves as a valuable proxy for air temperature in urban 
environments. LST represents the temperature of the Earth's surface as measured from satellite-based sensors, and it offers an excellent substitute for air temperature in urban 
settings due to the close relationship between land surface and atmospheric temperatures.Thereby, the ECOSTRESS (Ecosystem Spaceborne Thermal Radiometer Experiment on Space Station) 
mission and the related LST data emerges as a powerful tool to detect and understand the impact of heatwaves in urban settings.
"""

md_ecostress_data = """
We access the ECOSTRESS in swaths via the USGS API. The swath itself is a rectangular area of the earth's surface that is covered by the satellite's sensor and has a typical size of 400 km x 400 km.
The swaths itself have a spatial resolution of 70m x 70m. The data from the year 2022, which is the data that we use, is only available in the raw format, therefore we havbe to process it first to 
create geoTIFF files. For each swath we access three diffrent data types:
"""

md_swath_to_grid = """
In order to be able to project the data correctly on our map with the WGS84 projection, we have to perform a swath to grid transformation. This transformation is also stronlgy recommended by the USGS.
For the implementaiton of the transformation we mainly used the python package pyproj and GDAL. To implement the transformation we follow the steps described in the following:

Firstly, the transformation begins with obtaining the latitude-longitude bounding box of the satellite swath. This defines the geographic area covered by the data and serves as a reference for subsequent processing.
Next, the width and height of the bounding box are measured in the Azimuthal equidistant projection (AEQD). This projection is chosen for its ability to preserve area, ensuring accurate spatial 
relationships during subsequent calculations. Width and height are then divided by the desired pixel size to obtain the number of columns and rows in the final projection. With the AEQD projection established, a grid 
is created based on the desired number of columns and rows in the final projection, typically using the widely-used World Geodetic System (WGS 84). This structured grid forms the foundation for organizing the temperature 
data systematically. To achieve uniformity and consistency, the pixel size is adapted to create squared pixels. This adjustment simplifies computations and standardizes the representation of data across the grid.The final 
step involves mapping the temperature data from the original swath onto the newly created grid. K-D (K-dimensional) nearest neighbor resampling is employed for this purpose. Through K-D nearest neighbor resampling, each 
grid cell receives temperature values based on its proximity to the corresponding points in the original swath data. This process ensures that the temperature data is accurately represented at each grid cell, forming a 
comprehensive and precise gridded dataset.
"""

md_ecostress_data_quality = """
To ensure precise and high quality land surface temperature data, we came up with some various quality measures. Even when the data quality is observed easily by the human eye, this of course isnt applicable for a large
amount of data. Among others, the key quality measure is the cloud coverage. The cloud coverage is, as described above, a binary mask for each pixel indicating whether the pixel is covered by clouds. The cloud coverage is 
also provided by the USGS. The threshold for the cloud coverage is set to 25%, meaning that if all pixles in the defined area are covered by clouds, the data is not used for further analysis. The following pictures show the
cloud coverage for two different swaths. The first one is a swath with a cloud coverage of 36.18 % and the second one is a swath with a cloud coverage of 50.07 %. The white ares you can see in the LST pictures are the areas
with values that already exceed or are below certain temperature thresholds. This strongly correlates with the cloud coverage but doesn't apply for all areas. These areas are also not used for further analysis.
"""

md_ecostress_lst = """

"""


# <img src='https://github.com/MGenschow/DS_Project/blob/main/figures/grid_element_all.JPG?raw=true' width='70%' />
md_slx_1 = """
Having now both granular temperature data and classified land cover and land use characteristics availabe, we are now armed with data that allows us to proceed to the modeling phase.
The purpose of this phase is to establish a causal relationship between these features and the observed urban heat intensity.
As a first step, the data needs to be disaggregated to have some distinct observations to work with.
We do this by laying a grid over the city of Munich and its surrounding aggregating the data within each grid cell.
Note that this is not an uncommon approach in spatial econometrics (Kikon et al., 2016).
This allows us to have a sample of 8,528 observations to work with, while still retaining the spatial structure of the data.
The value of our dependent variable is calculated using a weighted average of all pixels that fall within a grid cell.
The independent variables are calculated by the share of surface that is covered by a certain LULC class.
The following chart visualizes this process:
"""

md_slx_2 = """
But how can we now use our sample to estimate the causal effect of our LULC characteristics on urban heat intensity (measured by the LST)?
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

- `Y` is the dependent variable (here: LST))
- `X` is a matrix of our explanatory variables (here: LULC characteristics)
- `W` is a spatial weight matrix
- `WX` represents spatially lagged independent variables

The inclusion of the term `WXγ` allows us to account for spatial spillover effects, i.e., how the land use characteristics of neighboring areas influence the temperature of the area in question.
For example, a large water body in a neighboring region might influence the temperature of our area of interest.
In practice, the lagged term consists of the the sum of the independent variables of all neighboring areas where neighboring areas refer to all grid elements that enclose the grid of interest.
What we mean as neighbouring areas can also be nicely seen from the chart above (refering to the orange surrounded part of the grid).

To account for important interactions between the features (building, low vegetation, water, trees and roads), we interacted them in `X`.
Additionally, we apply a log transformation to all of our independent variables to allow for decreasing marginal returns.
Lastly, we include average building height which we als calculated from data of the cadastral office as a proxy for the urban canyon effect.

The model parameter can then be straight-forwardly estimated using ordinary least squares.
The following table represents the full regression results:
"""

md_slx_3 = """
First of all, it is noteworthy that we achieve a very high model fit (i.e. R squared of 0.8).
Except the previously mentioned average height proxy and the water road interaction, all coefficients are statistically significant.
Due to high correlation (or multicollinearity) of the features and the many interactions involved, one should be cautious when interpreting the coefficients.
As a more intuitive guideline, we calculate average marginal effects that compare changes in OLS predictions across the whole sample
while accounting for the fact that a change in one feature is associated with simultaneous changes in the other features.

Generally, the results are in line with past literature and with what one would expect intuitively.
Water has the largest effect in absolute terms.
An increase in the share of water, trees or low vegetation is associated with a decrease in temperature, while buildings and roads are positively correlated with the dependent variable.

So much for the technical project background.
If you are now eager to see our causal model in action, the [HeatMapper](/HeatMapper) page is the place to go. Have fun!
"""

md_segmentation_into = """
Drawing on existing literature pertaining to the drivers and mitigating elements of urban heat, we utilize land use and land cover (LULC) attributes to pinpoint key 
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
#### Segmentation Pipeline
In order to achieve the high precision evident in our classification model, primarily featured in this application, we employ cutting-edge deep learning techniques 
from the realm of computer vision. These deep neural networks necessitate extensive data for training, and since no labeled data for Munich was accessible to us, 
we opted for a method known as transfer learning. This technique involves utilizing a model designed for one task and repurposing it for another task. By employing 
this method, we can train a larger model with the available data and then fine-tune it to cater to a specific task for which minimal data is available. In our context, 
we developed a training pipeline that incorporates a two-stage transfer learning approach.

After evaluating various model architectures, we decided on a DeepLabV3 (Chen, 2017) segmentation model with a ResNet101 backbone. The ResNet101 backbone is initialized with 
weights that have been trained on COCO.
In the first stage, we fine-tuned our model utilizing the Land-cOVEr Domain Adaptive semantic segmentation (LoveDA) (Wang, 2021) dataset, comprising 5987 high spatial resolution 
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

md_segmentation_results = """
#### Results
To assess whether the finetuning of the model actually improved our model performance, we composed a test set from our manually labelled images of Munich. We then compared
the accuracy of the model on the test set after the pretraining stage to the accuracy after the finetuning stage. As can be seen in the table below, finetuning the model led
an increase in the overall pixel-wise accuracy of ~9 percentage points and an increase in the mean accuracy of ~11 percentage points. The most important improvement could
be seen in the accuracy of the low vegetation class, which increased by 33 percentage points. This is especially important, as low vegetation is a key factor in mitigating 
urban heat. Additionally, also the accuracy of the building class and the accuracy of trees increased while the accuracy of impervious surfaces slightly decreased. As impervious 
surface is considered more of a "background" class, we do not consider this a major drawback.
"""

table_results = """
| Category          | Pretraining | Finetuning | Improvement |
|-------------------|------------------|------------------|-------------|
| Impervious        | 80.49            | 72.67            | -7.82       |
| Building          | 64.17            | 71.48            | 7.31        |
| Low Vegetation    | 43.71            | 77.22            | 33.51       |
| Trees             | 78.55            | 88.07            | 9.52        |
|-------------------|------------------|------------------|-------------|
| **Mean Accuracy**     | 66.73            | 77.36            | 10.63       |
| **Average Accuracy**  | 69.78            | 78.67            | 8.89        |
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
                            html.Li(html.A('Extracting land usage and land cover data from orthophotos', href='#section-2')),
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
                dcc.Markdown(md_hw_cit, style={"text-align": "justify"}, dangerously_allow_html=True)
            ], 
            className="mx-5 mb-4"
        ),
        dbc.Row(
                [
                html.H3('The ECOSTRESS Data', id='subsection-1-2'),
                dbc.Col(
                    [
                        dcc.Markdown(md_ecostress_intro, style={"text-align": "justify"}),
                    ],
                    width={'size': 6, 'offset': 0},
                ),
                dbc.Col(
                    [
                        html.Img(src=img_swath, style={'width': '100%'}),
                    ],
                    width={'size': 6, 'offset': 0},
                ),
                html.H3('The ECOSTRESS Data', id='subsection-1-2'),
                dcc.Markdown(md_ecostress_data, style={"text-align": "justify"}, dangerously_allow_html=True),
                html.Div(
                    [
                        html.Div(
                                [
                                html.Img(src=img_geo, style={'width': '100%'}),
                                html.P('GEO data which contains a latitude and longitude reference for each pixel')
                                ],
                            style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '10px'}
                            ),
                        html.Div(
                                [
                                html.Img(src=img_lst, style={'width': '100%'}),
                                html.P('LST data which contains the actual land surface temperature for each pixel')
                                ],
                            style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '10px'}
                            ),
                        html.Div(
                                [
                                html.Img(src=img_cloud, style={'width': '100%'}),
                                html.P('A binary cloud mask that indicates whether the pixel is covered by clouds')
                                ],
                            style={'width': '30%', 'display': 'inline-block', 'text-align': 'center', 'margin': '10px'}
                            )
                            ],
                            style={'width': '100%'}
                            ),
                html.H3('Swath-to-grid Transformation', id='subsection-1-3'),
                dcc.Markdown(md_swath_to_grid, style={"text-align": "justify"}),
                html.H3('Data Quality Measurement', id='subsection-1-4'),
                dcc.Markdown(md_ecostress_data_quality, style={"text-align": "justify"}),
                dbc.Col(
                    [   html.H3("Cloudcoverage 36.18%:"),
                        dbc.Carousel(
                            id="carousel_comp_1",
                            items=[
                                {
                                    "key": "1",
                                    "src": img_QC_LST_1,
                                    "header": "",
                                    "caption": "",
                                    "imgClassName": "carousel-image"
                                },
                                {
                                    "key": "2",
                                    "src": img_QC_CLD_1,
                                    "header": "",
                                    "caption": "",
                                    "imgClassName": "carousel-image"
                                },
                            ],
                            controls=False,
                            indicators=False,
                            interval=None,
                        ),
                        dbc.RadioItems(
                            id="comp-number_1",
                            options=[
                                {"label": "LST", "value": 0},
                                {"label": "Cloud", "value": 1}
                                ],
                            value=0,
                            inline=True,
                            )
                    ], width = {'size':6, 'offset': 0, 'md':'auto'},
                ),
                dbc.Col(
                    [
                        html.H3("Cloudcoverage 50.07%:"),
                        dbc.Carousel(
                            id="carousel_comp_2",
                            items=[
                                {
                                    "key": "1",
                                    "src": img_QC_LST_2,
                                    "header": "",
                                    "caption": "",
                                    "imgClassName": "carousel-image"
                                },
                                {
                                    "key": "2",
                                    "src": img_QC_CLD_2,
                                    "header": "",
                                    "caption": "",
                                    "imgClassName": "carousel-image"
                                },
                            ],
                            controls=False,
                            indicators=False,
                            interval=None,
                        ),
                        dbc.RadioItems(
                            id="comp-number_2",
                            options=[
                                {"label": "LST", "value": 0},
                                {"label": "Cloud", "value": 1}
                                ],
                            value=0,
                            inline=True,
                            )
                    ], width = {'size':6, 'offset': 0, 'md':'auto'},
                ),
                html.H3('Observing Land Surface Temperature', id='subsection-1-5'),
                dcc.Markdown(md_ecostress_lst, style={"text-align": "justify"}),
            ], 
            className="mx-5 mb-4"
        ),

        dbc.Row(
            [
                html.H2('Extracting land usage and land cover data from orthophotos', id='section-2'),
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
                html.Img(src='assets/segmentation_pipeline.png', alt='Segmentation model', width='70%'),
            ],
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                dcc.Markdown(md_segmentation_results, style={"text-align": "juytify"})
            ],
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                html.Div(dcc.Markdown(table_results), style={"margin": "0 auto", "max-width": "800px"})
            ],
            className="mx-5 mb-4"
        ),
        dbc.Row(
            [
                html.H2('Econometrics: Modeling LST using a SLX model', id='section-3'),
                dcc.Markdown(md_slx_1, style={"text-align": "justify"}),
                html.Div(
                    [
                    html.Img(src='assets/disaggregation.jpg', alt='Disaggregation chart', width='520px'),
                    ],
                    style={'text-align': 'center'}
                ),
                dcc.Markdown(md_slx_2, style={"text-align": "justify"}),
                html.Div(
                    [
                    html.Img(src='assets/slx_results.jpg', alt='SLX results', width='480px'),
                    ],
                    style={'text-align': 'center'}
                ),
                dcc.Markdown(md_slx_3, style={"text-align": "justify"}),
                html.H2('References', id='section-ref'),
                dcc.Markdown(md_ref),
            ], 
            className="mx-5 mb-4"
        ),
        html.Div(style={'height': '15vh'})
    ],
    style={'height': '100vh', 'overflowY': 'scroll'},
    fluid=True,
    className="m-1"
)



@callback(
    Output("carousel_comp_1", "active_index"),
    Input("comp-number_1", "value"),
)
def select_slide(idx):
    return idx

@callback(
    Output("carousel_comp_2", "active_index"),
    Input("comp-number_2", "value"),
)
def select_slide(idx):
    return idx