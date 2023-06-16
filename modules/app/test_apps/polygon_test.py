import dash
import dash_leaflet as dl
from dash import dcc, html, Input, Output, State
import geopandas as gpd
from shapely.geometry import Polygon

app = dash.Dash()

# Create a Polygon geometry
polygon = Polygon([(11.574, 48.135), (11.584, 48.135), (11.584, 48.145), (11.574, 48.145), (11.574, 48.135)])

# Create a GeoDataFrame with the Polygon
polygon_data = {"ID": [1], "geometry": [polygon]}
gdf = gpd.GeoDataFrame(polygon_data, crs="EPSG:4326")
print(gdf)
# Convert the Polygon to GeoJSON
geojson = gdf.to_json()

app.layout = html.Div(
    [
        dl.Map(
            [
                dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                dl.GeoJSON(
                    data=geojson,
                    id="polygon",
                    options={"style": {"color": "blue", "fillOpacity": 0.4}},
                    zoomToBoundsOnClick=True,
                    children=[dl.Popup("Polygon", className="popup")],
                ),
            ],
            id="map",
            center=(48.139, 11.572),
            zoom=13,
            style={"width": "100%", "height": "400px"},
        ),
        html.Div(id="polygon-id-output"),
    ]
)


@app.callback(
    Output("polygon-id-output", "children"),
    [Input("polygon", "n_clicks")],
)
def display_polygon_id(n_clicks):
    if n_clicks is not None:
        polygon_id = gdf.loc[0, "ID"]
        return f"Polygon ID: {polygon_id}"
    else:
        return ""

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
