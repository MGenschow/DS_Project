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
import base64
from PIL import Image
from io import BytesIO
import json


# Import functions from other files 
from information import information_tab
from footer import footer
from navbar import navbar

# Build the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
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
data = json.load(open("modules/app/data/gemeinden_simplify200.geojson", "r"))

# Load the features.json file
features = dl.GeoJSON(data=data, zoomToBoundsOnClick=True, id="geojson")


# --------------------
# SIDEBAR STYLE
# --------------------

# Define the style for the sidebar and the content
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 55,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#456789",
    "zIndex": 1,  # Set the sidebar above the content
}

# Padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "zIndex": 0,  # Set the content below the sidebar
}

# Define the sidebar
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("Menu", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/information", active="exact"),
                dbc.NavLink(
                    "Temperature comparison", href="/temperature", active="exact"
                ),
                dbc.NavLink(
                    "Heat map with clustering", href="/map", active="exact"
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Define the content layout
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# App layout
app.layout = html.Div([dcc.Location(id="url"), navbar, sidebar, content, footer])

# Callback Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Define the function to render the page content
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # First tab
    if pathname == "/information":
        return [information_tab]
    # Second tab
    elif pathname == "/temperature":
        return [temperature_overview]
    # Third tab
    elif pathname == "/map":
        return [
            html.Div(
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
        ]

    return [
        html.Div(
            dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognized..."),
                ]
            )
        )
    ]


@app.callback(
    [Output("text-output", "children"), Output("image-container", "children")],
    [Input("geojson", "click_feature")],
)
def update_output(click_feature):
    if click_feature is not None:
        properties = click_feature["properties"]
        destatis = properties.get("destatis", {})
        population_density = destatis.get("population_density", "")

        if population_density:
            image_path = f"modules/app/data/{population_density}.tif"
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
                None,
            )
    else:
        return "Click on a district to get its population density.", None








# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8081, use_reloader=False)
