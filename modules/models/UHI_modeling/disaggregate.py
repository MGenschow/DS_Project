#%% import packages
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import folium
import folium.raster_layers
import folium.features
import rasterio
import rioxarray as rxr

from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon
from shapely.geometry import box
from branca.colormap import LinearColormap

home_directory = os.path.expanduser( '~' )
os.chdir(home_directory + '/DS_Project/modules')
config_path = 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
#%% haversine
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the Earth using the Haversine formula.

    Args:
        lon1 (float): Longitude of the first point in decimal degrees.
        lat1 (float): Latitude of the first point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.

    Returns:
        float: The distance between the two points in kilometers.

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
#%% convert_lon_lat_to_lat_lon
def convert_lon_lat_to_lat_lon(point):
    """
    Convert a point in longitude and latitude format to latitude and longitude format.

    Args:
        point (list): A list containing the longitude and latitude coordinates in decimal degrees.

    Returns:
        list: A list containing the latitude and longitude coordinates in decimal degrees.

    """
    return [point[1], point[0]]
#%%
def convert_bbox_lon_lat_to_lat_lon(bbox):
    """
    Convert a bounding box in longitude and latitude format to latitude and longitude format.

    Args:
        bbox (list): A list containing the bounding box coordinates in the order [lon1, lat1, lon2, lat2].

    Returns:
        list: A list containing the converted bounding box coordinates in the order [lat1, lon1, lat2, lon2].

    """
    # Convert the first two coordinates (lon1, lat1)
    converted_bbox = [convert_lon_lat_to_lat_lon(bbox[:2])]

    # Convert the last two coordinates (lon2, lat2)
    converted_bbox.append(convert_lon_lat_to_lat_lon(bbox[2:]))

    # Flatten the converted_bbox list
    return [coord for point in converted_bbox for coord in point]
#%% divide_polygon_into_grid
def divide_polygon_into_grid(polygon, grid_size_meters):
    """
    Divide a polygon into a grid of smaller polygons.

    Args:
        polygon (Polygon): The input polygon to be divided.
        grid_size_meters (float): The size of each grid cell in meters.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the grid polygons.

    """
    # Calculate the longitudinal and latitudinal distances of the polygon
    lon_distance = polygon.bounds[2] - polygon.bounds[0]
    lat_distance = polygon.bounds[3] - polygon.bounds[1]
    
    # Calculate the grid size in terms of longitudinal and latitudinal units
    lon_grid_size = grid_size_meters / (111320 * math.cos(math.radians(polygon.bounds[1])))
    lat_grid_size = grid_size_meters / 111320
    
    # Initialize a list to store the grid polygons
    grid_polygons = []
    
    # Iterate over the range of longitudinal and latitudinal values to create grid polygons
    for lon in np.arange(polygon.bounds[0], polygon.bounds[2], lon_grid_size):
        for lat in np.arange(polygon.bounds[1], polygon.bounds[3], lat_grid_size):
            # Create a square polygon using the current coordinates and grid sizes
            square = Polygon([
                (lon, lat), (lon + lon_grid_size, lat),
                (lon + lon_grid_size, lat + lat_grid_size), (lon, lat + lat_grid_size)
            ])
            
            # Check if the polygon intersects with the current square
            if polygon.intersects(square):
                intersection = polygon.intersection(square)
                
                # Check if the intersection area is greater than zero
                if intersection.area > 0:
                    # Handle different intersection geometry types
                    if intersection.geom_type == 'Polygon':
                        if intersection.equals(square):
                            grid_polygons.append(intersection)
                    elif intersection.geom_type == 'MultiPolygon':
                        for part in intersection:
                            if part.equals(square):
                                grid_polygons.append(part)
    
    # Create a GeoDataFrame from the grid polygons
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons)
    grid_gdf.crs = 'EPSG:4326'
    
    return grid_gdf
#%% pixels_to_foliumMap
def pixels_to_foliumMap(array, polygon):
    """
    Convert pixel array data to a Folium map with overlays.

    Args:
        array (ndarray): The pixel array data.
        polygon (Polygon): The polygon representing the bounds of the pixel array.

    Returns:
        folium.Map: A Folium map with overlays.

    """
    # Extract the pixel array data
    data = np.array(array)[0]

    # Define color range for colormap
    color_range = np.linspace(0, 1, 256)
    colors_jet_rgba = plt.cm.jet(color_range)
    colors_jet_hex = [mcolors.rgb2hex(color) for color in colors_jet_rgba]

    # Create colormap and normalize the data
    cmap = plt.colormaps['jet']
    norm = colors.Normalize(vmin=data.min(), vmax=data.max())
    colored_data = cmap(norm(data))

    # Calculate image bounds and corner coordinates
    image_bounds = box(*array.rio.bounds())
    min_x, min_y, max_x, max_y = array.rio.bounds()
    corner_coordinates = [[min_y, min_x], [max_y, max_x]]

    # Create a Folium map centered on the image bounds
    m = folium.Map(
        location=[image_bounds.centroid.y, image_bounds.centroid.x],
        zoom_start=15,
    )

    # Add a GeoJson overlay for the image bounds
    folium.GeoJson(
        image_bounds.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'orange', 'colorOpacity': 0.7}
    ).add_to(m)

    # Add a tile layer to the map
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='CartoDB',
        transparent=True,
    ).add_to(m)

    # Add the colored data as an ImageOverlay to the map
    folium.raster_layers.ImageOverlay(
        colored_data,
        bounds=corner_coordinates,
        opacity=0.5,
        interactive=True,
        cross_origin=True,
        pixelated=True,
        zindex=0.2
    ).add_to(m)

    # Create a colormap for the data and add it to the map
    colormap = LinearColormap(
        colors=colors_jet_hex,
        vmin=data.min(),
        vmax=data.max(),
        max_labels=15
    )
    colormap.caption = 'Land surface temperature in celsius'
    colormap.add_to(m)

    # Add a GeoJson overlay for the polygon
    folium.GeoJson(
        polygon.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'black', 'colorOpacity': 0.7}
    ).add_to(m)

    return m
#%% naive_pixel_mean
def naive_pixel_mean(array, polygon):
    """
    Calculate the naive pixel mean within a polygon.

    Args:
        array (rasterio.io.DatasetReader): The raster dataset.
        polygon (Polygon): The polygon to calculate the mean within.

    Returns:
        float: The mean pixel value within the polygon.

    """
    # Clip the raster dataset using the polygon
    c = array.rio.clip([polygon], crs=4326, all_touched=True)

    # Calculate the mean of the clipped pixels
    m = c.mean().values

    return m
#%% rec_polygon_coords
def rec_polygon_coords(polygon):
    """
    Retrieve the corner coordinates of a rectangle polygon.

    Args:
        polygon (Polygon): The rectangle polygon.

    Returns:
        tuple: A tuple containing the minimum x-coordinate, minimum y-coordinate,
               maximum x-coordinate, and maximum y-coordinate of the rectangle.

    """
    # Retrieve the points of the polygon's exterior
    points = list(polygon.exterior.coords)

    # Separate the x and y coordinates
    x_coordinates, y_coordinates = zip(*points)

    # Return the corner coordinates of the rectangle
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)