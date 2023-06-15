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

# Load login credentials and the config file
# Import credentials from credentials.yml file
try:
    with open('/home/tu/tu_tu/' + os.getcwd().split('/')[6] +'/DS_Project/modules/credentials.yml', 'r') as file:
        credentials = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: Credentials file not found. Please make sure the file exists.")

# Import bounding boxes for Munich from config.yml file
with open('/home/tu/tu_tu/' + os.getcwd().split('/')[6] + '/DS_Project/modules/config.yml', 'r') as file: 
    config = yaml.safe_load(file)

# Set login parameters (USGS/ERS login)
login_ERS = {
    'username' : credentials["username"],
    'password' : credentials["password_ERS"]
    }

# Request and store token
# Request token
response = requests.post(config['api']['path'] + 'login', data=json.dumps(login_ERS))

# Set header
headers = {'X-Auth-Token': response.json()['data']}

# Import heatwaves
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')

# Combine dates to periods format of heatwaves
heatwaves = heatwave_transform(dates)
# Add heatwaves from 2021
heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-22 00:00:00'})
heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-16 00:00:00'})

# Import tropical timeperiods
tropicalDays = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/tropicalPeriods.pkl')

# Combine both
hW_tD = heatwaves + tropicalDays

# Invert heatwave
import datetime

start_date = datetime.date(2022, 6, 1)
end_date = datetime.date(2022, 8, 31)
date_list = []

delta = datetime.timedelta(days=1)
current_date = start_date

while current_date <= end_date:
    date_element = {
        'start': current_date.strftime('%Y-%m-%d 00:00:00'),
        'end': (current_date + delta).strftime('%Y-%m-%d 00:00:00')
    }
    date_list.append(date_element)
    current_date += delta


inverted_HW = []

# 
for dates in date_list:
    if not dateInHeatwave(datetime.datetime.strptime(dates['start'],'%Y-%m-%d %H:%M:%S'),hW_tD):
        inverted_HW.append(dates)
    elif not dateInHeatwave(datetime.datetime.strptime(dates['end'],'%Y-%m-%d %H:%M:%S'),hW_tD):
        inverted_HW.append(dates)



# Set spatial filter;
spatialFilter =  {
    'filterType' : "mbr",
    'lowerLeft' : {
        'latitude' : config["bboxes"]["munich"][1],
        'longitude' : config["bboxes"]["munich"][0]
        },
    'upperRight' : {
        'latitude' : config["bboxes"]["munich"][3],
        'longitude' : config["bboxes"]["munich"][2]
        }
    }
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
    {'start': '2022-03-01 00:00:00', 'end': '2022-06-02 00:00:00'},
    {'start': '2022-08-30 00:00:00', 'end': '2022-12-01 00:00:00'}]


months = split_window(month, window_size=10, overlap=1)



# 
for temporalFilter in months:
    downloadH5(credentials, headers, temporalFilter, spatialFilter, config)




