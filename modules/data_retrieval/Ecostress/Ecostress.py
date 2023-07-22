# %% Import packages and load config file
import requests
import json
#from io import BytesIO
#import rasterio
#import rioxarray
#import h5py
#import subprocess 
#import numpy as np
#import matplotlib.pyplot as plt
#from pyresample import geometry as geom 
#from pyresample import kd_tree as kdt
#import pyproj
#from rasterio.plot import show
#from osgeo import gdal, gdal_array, gdalconst, osr
#from PIL import Image
#import os
import yaml
from os import listdir
from os.path import isfile, join
import datetime
#import pandas as pd
#from shapely.geometry import box
#import folium
#import matplotlib.colors as colors
#from rasterio.enums import Resampling
#import tifffile
import imageio
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


# %% Create header for the API call
# Set login parameters (USGS/ERS login)
login_ERS = {
    'username' : credentials["username"],
    'password' : credentials["password_ERS"]
    }

# Request and store token
response = requests.post(config['api']['path'] + 'login', data=json.dumps(login_ERS))

# Set header
headers = {'X-Auth-Token': response.json()['data']}


# %% Import heatwaves
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')


# Transform heatwaves
heatwaves = heatwave_transform(dates)

# Import tropical timeperiods
# tropicalDays = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/tropicalPeriods.pkl')

# Combine heatwaves and heatdays
# hW_tD = heatwaves + tropicalDays

#  Invert the heatwave
inverted_HW = invertHeatwave(heatwaves)


# Calculate the temperature range for each month
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')
tempRange = calculateTempRange(dwd)
    

# %% Set spatial filter for API Call
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


# %% Download all files corresponding to the heatwaves
# NOTE: Can take, depending on the parameters, quite some time 
# (up to several hours)

# Specify period to loop over
month = [{'start': '2022-03-01 00:00:00', 'end': '2022-06-05 00:00:00'}]

confirmation = input("Do you want to download the hierarchical files (Y/n): ")
if confirmation.lower() == "y":

    for temporalFilter in month:
        downloadH5(credentials, headers, temporalFilter, spatialFilter, config)
else:
    print("Loop execution cancelled.")


# %% QC
# Count the number of files for a specific period
month = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]
types = ['GEO','CLOUD', 'LSTE']
# Loop over files
for t in types: 
    files =  [
        f
        for f in listdir(config['data']['ES_raw']) 
        if t in f and dateInHeatwave(datetime.strptime(f.split('_')[5], '%Y%m%dT%H%M%S'), month)
        ]
        
    print(f'There are {len(files)} {t} files')


# %% Create a tiff for each unique scene
# Define period to create tiffs
period = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]

confirmation = input("Do you want to create tiffs for hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    processHF(period, tempRange, config)
else:
    print("Function execution cancelled.")


# %% QC
# Check if each LSTE.h5 has a respective tif

# Extract keys
keys = set([
    files.split('_')[3] + '_' + files.split('_')[4]
    for files in listdir(config['data']['ES_raw']) if 'LSTE' in files
    ])
# Extract tiff paths
tiffs = [
    f 
    for f in listdir(config['data']['ES_tiffs'])
    if isfile(join(config['data']['ES_tiffs'], f)) and f.endswith('.tif')
    ]
# Loop over keys
for k in keys:
    l = len([f for f in tiffs if k in f])

    if l < 1:
        print(f'There are no tiff files for the key {k}')
        print([f for f in raw_files if k in f])


# %% 
dataQ = dataQualityOverview(inverted_HW,0,15,config)

# %% Get LST and Cloud by orbit key
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

key = '22241_007'

files = [f for f in listdir(config['data']['ES_tiffs']) if key in f]


path = config['data']['ES_tiffs'] + files[2]

# %%
# Read the image using matplotlib.image.imread
img = mpimg.imread(path)

# Plot the image
plt.imshow(img)
plt.axis('off')  # Turn off axis ticks and labels
plt.show()

# %%
# Read the TIFF image using rasterio
with rasterio.open(path) as src:
    # Read the image data (bands) as a NumPy array
    img = src.read()

# Plot the image
plt.imshow(img[0])  # Use 'viridis' colormap for better visualization
#plt.colorbar()  # Add a colorbar to the plot
plt.axis('off')  # Turn off axis ticks and labels
plt.show()


# %% Create mean tiffs for inside and outside the heatwaves for 
# Morning and afternoon 
meanTiff(heatwaves, insideHeatwave=True, config=config)
meanTiff(inverted_HW, insideHeatwave=False, config=config)

# %% Plot the respective tiff
tiff_path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/avgAfterNoon_HW.tif'
png_path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/avgAfterNoon_HW.png'
tif = rasterio.open(path)

plt.imshow(tif.read()[0],'jet')
#plt.colorbar(label = "Temperature in Celsius")
plt.show()
# %%
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Read the TIFF file using rasterio
with rasterio.open(tiff_path) as tif:
    # Read the data from the TIFF file
    tiff_data = tif.read(1)  # Read the first band (assuming it's a single-band TIFF)

# Normalize the data to [0, 1] range
data_min = np.min(tiff_data)
data_max = np.max(tiff_data)
normalized_data = (tiff_data - data_min) / (data_max - data_min)

# Apply the 'jet' colormap
cmap = cm.get_cmap('jet')
colored_data = cmap(normalized_data)

# Save the colored data as a PNG file using imageio
imageio.imwrite(png_path, (colored_data * 255).astype(np.uint8))

# %% Crop the tif to closer boundig boxes
from rasterio.mask import mask
xmin, ymin, xmax, ymax = config['bboxes']['munich_grid']
bbox = box(xmin, ymin, xmax, ymax)

# Crop the GeoTIFF to the bounding box
cropped, transform = mask(dataset=tif, shapes=[bbox], crop=True)

plt.imshow(cropped[0],'jet')
plt.colorbar(label = "Temperature in Celsius")
plt.show()


# %%
map_afternoon = tiffs_to_foliumMap(
    path,
    pixelated=False,
    minTemp=tif.read()[0].min(),
    maxTemp=tif.read()[0].max()
    )

map_afternoon


# %% Plot tif over a interactive open street map
rootpath = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/'

files = {
    'morning':[
        'avgMorning_nonHW.tif',
        'avgMorning_HW.tif'
        ],
    'afterNoon':[
        'avgAfterNoon_nonHW.tif',
        'avgAfterNoon_HW.tif'
        ]
}

for f in files:
    
    tMin, tMax = 99, -99

    for t in files[f]:
        tif = rasterio.open(os.path.join(rootpath, t))

        if tMin > np.quantile(tif.read()[0],0.001):
            tMin = np.quantile(tif.read()[0],0.001)

        if tMax < np.quantile(tif.read()[0],0.999):
            tMax = np.quantile(tif.read()[0],0.999)
    
    for t in files[f]:
        map = tiffs_to_foliumMap(os.path.join(rootpath, t), pixelated=False, minTemp=tMin, maxTemp=tMax)
        map.save(t.replace('.tif','.html'))


# %%
#map_afternoon = tiffs_to_foliumMap(os.path.join(rootpath, files[0]), pixelated=False)
#map_afternoon

# Save map
# map_afternoon.save('afterNoon_nonHT.html')


# %% Create a data quality overview for a selected period
dataQ = dataQualityOverview(inverted_HW, 0, 25, config)

# %% Create DF with relevant tifs for the respective period
dataQ_selected = dataQ[
    # Morning: 5-8, Afternoon: 12-16
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 12 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 16) & 
    dataQ['qualityFlag']
    ]
# %% Plot all relevant tiffd
for o in dataQ_selected.orbitNumber:
    plot_by_key(o,'LSTE', config)
    plot_by_key(o,'Cloud', config)


# %% Create a tiff with the difference of Afternoon and Morning
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/'
morningHW = rasterio.open(path + 'avgMorning_HW.tif')
afternoonHW = rasterio.open(path +'avgAfterNoon_HW.tif')

# Calculate the difference
diff_NoonMorning = afternoonHW.read()[0] - morningHW.read()[0]
#diff_NoonMorning[diff_NoonMorning < 0 ] = np.NaN

# Plot the difference
plt.imshow(diff_NoonMorning,'jet_r', vmin=0, vmax=19)
plt.colorbar(label = "Temperature in Celsius")

plt.show()

