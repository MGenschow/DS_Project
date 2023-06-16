import dash
import dash_leaflet as dl
from dash import dcc, html, Input, Output, State
import json

app = dash.Dash()

# Create a text output that displays the population density
text_output = html.Div(id="text-output")

attribution = (
    'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors,'
    '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'
)

url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

# import geojson data
data = json.load(open("modules/app/data/gemeinden_simplify200.geojson", "r"))

# Load the features.json file
features = dl.GeoJSON(
    data=data,
    zoomToBoundsOnClick=True,
    id="geojson",
    options=dict(
        style=dict(color="green"),
        fillColor="green",
        highlightStyle=dict(weight=2, color="#666", dashArray=""),
    ),
)

app.layout = html.Div(
    [
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
            id="map",
            style={
                "width": "100%",
                "height": "50vh",
                "margin": "auto",
                "display": "block",
                "position": "relative",
            },
        ),
        html.Br(),
        text_output,
    ]
)


@app.callback(
    Output("text-output", "children"),
    [Input("geojson", "click_feature")],
)
def update_text_output(click_feature):
    if click_feature is not None:
        properties = click_feature["properties"]
        destatis = properties.get("destatis", {})
        population_density = destatis.get("population_density", "")
        return f"Population Density: {population_density} people per square kilometer"
    else:
        return "Click on an area to get its population density"


if __name__ == "__main__":
    app.run_server(debug=True, port=8031)
