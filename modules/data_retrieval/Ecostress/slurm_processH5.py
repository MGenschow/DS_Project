# Import packages
import requests
import json
from io import BytesIO
import rasterio
import rioxarray
import h5py
import subprocess 
import numpy as np
import matplotlib.pyplot as plt
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
from os.path import join
import pyproj
from rasterio.plot import show
from osgeo import gdal, gdal_array, gdalconst, osr
from PIL import Image
import os
import yaml
from os import listdir
from os.path import isfile, join
import pandas as pd
from shapely.geometry import box
import folium
import matplotlib.colors as colors
from rasterio.enums import Resampling

# Import all functions from utils
from utils import *

#  Load login credentials and the config file
# Import credentials from credentials.yml file
try:
    with open('/home/tu/tu_tu/' + os.getcwd().split('/')[6] +'/DS_Project/modules/credentials.yml', 'r') as file:
        credentials = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: Credentials file not found. Please make sure the file exists.")

# Import bounding boxes for Munich from config.yml file
with open('/home/tu/tu_tu/' + os.getcwd().split('/')[6] + '/DS_Project/modules/config.yml', 'r') as file: 
    config = yaml.safe_load(file)

from datetime import datetime, timedelta


def split_window(window, window_size, overlap):
    result = []
    for window in window:
        start_date = datetime.strptime(window['start'], '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(window['end'], '%Y-%m-%d %H:%M:%S')
        current_date = start_date

        while current_date + timedelta(days=window_size) <= end_date:
            sub_window = {
                'start': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'end': (current_date + timedelta(days=window_size)).strftime('%Y-%m-%d %H:%M:%S')
            }
            result.append(sub_window)
            current_date += timedelta(days=window_size - overlap)

    return result


# Download all files corresponding to the heatwaves
month = [
    {'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}
    ]


months = split_window(month, window_size=5, overlap=1)

# Loop over months
for period in months:
    processHF([period], config)