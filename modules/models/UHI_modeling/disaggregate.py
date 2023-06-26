#%% import packages
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import pickle
import yaml

from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon

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