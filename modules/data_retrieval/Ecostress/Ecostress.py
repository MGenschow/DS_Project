# %% Import packages
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
import statistics
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

# %% Set login parameters (USGS/ERS login)
login_ERS = {
    'username' : credentials["username"],
    'password' : credentials["password_ERS"]
    }

# %% Request and store token
# Request token
response = requests.post(config['api']['path'] + 'login', data=json.dumps(login_ERS))

# Set header
headers = {'X-Auth-Token': response.json()['data']}

# %% Import heatwaves
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')

# Combine dates to periods format of heatwaves
heatwaves = heatwave_transform(dates)
# Add heatwaves from 2021
heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-21 23:59:00'})
heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-15 23:59:00'})

# %% Set spatial filter;
spatialFilter =  {
    'filterType' : "mbr",
    'lowerLeft' : {
        'latitude' : config["bboxes"]["munich"][1],
        'longitude' : config["bboxes"]["munich"][0]
        },
        'upperRight' : {
            'latitude' : config["bboxes"]["munich"][3],
            'longitude' : config["bboxes"]["munich"][2]}
            }

# %% Download all files corresponding to the heatwaves
# NOTE: Can take, depending on the parameters, quite some time 
# (up to several hours)
confirmation = input("Do you want to download the hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    for temporalFilter in heatwaves:
        downloadH5(credentials, headers, temporalFilter, spatialFilter, config)
else:
    print("Loop execution cancelled.")

# %% Delete all existings tiffs and create a tiff for each unique scene
processHF(heatwaves, config)

# %% Create a Dataframe to check the quality of all relevant tiffs
dataQ = dataQualityOverview(config)

# %% Plot LST tiff by key
#key = '23129_012'
#lst = rioxarray.open_rasterio(
#    config['data']['ES_tiffs'] + 
#    [f for 
#    f in [p for p in onlyfiles if key in p] 
#    if 'LSTE' in f and '.tif' in f][0]
#    )
#img = np.array(lst)[0]
#plt.imshow(img, cmap='jet')
#plt.show()

# %% 
# Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 15 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 17) & 
    dataQ['qualityFlag']
    ]

# %%
# Store orbit numbers
orbitNumbers = afterNoon['orbitNumber']

# Calculate mean masked array
output = meanMaskArray(orbitNumbers, config)

# Store output in variables
mean_array, maskedArraysL, pixel_sizes, bounding_boxes = output

# %% Store mean array as tiff
# Set pixel size for geo tiff
pixel_size = [statistics.mean(values) for values in zip(*pixel_sizes)]

# Set values for the bounding box
left = np.mean([box.left for box in bounding_boxes])
bottom = np.mean([box.bottom for box in bounding_boxes])
right = np.mean([box.right for box in bounding_boxes])
top = np.mean([box.top for box in bounding_boxes])

# Create bounding box
boundingBox = rasterio.coords.BoundingBox(left, bottom, right, top)
xmin, ymin, xmax, ymax = boundingBox

# Set geotransform
geotransform = (xmin, pixel_size[0], 0, ymax, 0, pixel_size[1])

# Create geotiff
array_to_tiff(mean_array, 'mean_afternoon.tif' , geotransform)

# %% Plot tif
tif = rasterio.open('mean_afternoon.tif')
plt.imshow(tif.read()[0],'jet')

# %% Plot mean array
plt.imshow(mean_array,'jet')

# Count number of not matching values
n = mean_array.data.size-np.sum(mean_array.data == tif.read()[0])
print(f'{n} elements differ in both arrays')

# %% Create a subplot with all tiffs
# Get number of arrays
num_plots = len(maskedArraysL)
# Calculate number of rows and cols
rows = int(np.ceil(np.sqrt(num_plots)))
cols = int(np.ceil(num_plots / rows))

# Initiate subplot
fig, axs = plt.subplots(rows, cols)

# Loop over maskedArraysL
for i, ax in enumerate(axs.flat):
    if i < num_plots:
        ax.imshow(maskedArraysL[i], cmap='jet')
        ax.axis('off')
    else:
        ax.axis('off')

# Plot overall plot
plt.tight_layout()  
plt.show()


# %% Plot tif over a interactive open street map
array_to_foliumMap('mean_afternoon.tif')

# TODO: Plot the mean tiff
# TODO: Reduce map to munich 
# TODO: Add a legend
# TODO: Add water to the map

