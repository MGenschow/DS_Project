#%% import packages
import pandas as pd
import numpy as np
import math
import os
import pickle
import yaml

from math import radians, sin, cos, asin, sqrt

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
