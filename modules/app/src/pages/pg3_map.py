import dash
import dash_leaflet as dl
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, html, dash_table, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import dash_extensions as de
from dash_bootstrap_components import Container
import base64
from PIL import Image
from io import BytesIO
import json

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/Map',  # '/' is home page and it represents the url
                   name='Map with Heat information ',  # name of page, commonly used as name of link
                   title='Go fuck yourself',  # title that appears on browser's tab
                   #image='pg1.png',  # image in the assets folder
                   description='Final map of our project'
)

# Preparing the map
# Create a text output that returns the name of the input image when clicked
text_output = html.Div(
    id="text-output", children="Click on the district to get its population density."
)

# Create an empty image container
image_container = html.Div(id="image-container")

attribution = (
    'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors,'
    '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'
)

url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

# Import geojson data
data = json.load(open("/Users/skyfano/Documents/Data_Science_Project/DS_Project/modules/app/src/assets/gemeinden_simplify200.geojson", "r"))

# Load the features.json file
features = dl.GeoJSON(data=data, zoomToBoundsOnClick=True, id="geojson")


layout = html.Div(
                [
                    html.H3(
                        "Map of Munich showing the districts and their population density",
                        style={"text-align": "center"},
                    ),
                    dl.Map(
                        [
                            features,
                            dl.TileLayer(
                                url=url,
                                attribution=attribution,
                            ),
                        ],
                        center=[48.137154, 11.576124],
                        zoom=12,
                        id="map_2",
                        style={
                            "width": "100%",
                            "height": "600px",
                            "margin": "auto",
                            "display": "block",
                            "position": "relative",
                        },
                    ),
                    html.Br(),
                    text_output,
                    html.Br(),
                    image_container,
                ]
            )


# --------------------

@callback(
    [Output("text-output", "children"), Output("image-container", "children")],
    [Input("geojson", "click_feature")],
)
def update_output(click_feature):
    if click_feature is not None:
        properties = click_feature["properties"]
        destatis = properties.get("destatis", {})
        population_density = destatis.get("population_density", "")

        if population_density:
            image_path = f"/Users/skyfano/Documents/Data_Science_Project/DS_Project/modules/app/src/assets/{population_density}.tif"
            image = Image.open(image_path)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue())

            image_element = html.Img(
                src=f"data:image/png;base64,{encoded_image.decode()}",
                width="200px",
                height="200px",
            )
            return (
                f"Population Density: {population_density} people per square kilometer",
                image_element,
            )
        else:
            return (
                "Population density information not available for this district.",
                ' ',
            )
    else:
        return "Click on a district to get its population density.", None