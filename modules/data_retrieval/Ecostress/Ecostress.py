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

'''
# %% Create a loop that creates and stores a mean tif for each first and second half of a month

# Split period
period = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]
periods = split_period(period, split_half=False)

# %% Plot all 12 tiff files in one plot
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff/'

tiff_files = [file for file in os.listdir(path) if file.endswith('.tif')]
tiff_files.sort()

rows, cols = 4, 3

# Create the subplot
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

# Loop through the TIFF files and plot them in the subplot
for i, file in enumerate(tiff_files):
    row = i // cols
    col = i % cols
    tif = rasterio.open(os.path.join(path, file))
    axes[row, col].imshow(tif.read()[0],'jet') #, vmin=-15, vmax=45)
    axes[row, col].set_title(file)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the subplot
plt.show()
'''

# %% Create mean tiffs for inside and outside the heatwaves for 
# Morning and afternoon 
meanTiff(heatwaves, insideHeatwave=True, config=config)
meanTiff(inverted_HW, insideHeatwave=False, config=config)

# %% Plot the respective tiff
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/avgMorning_HW.tif'
tif = rasterio.open(path)

plt.imshow(tif.read()[0],'jet')
plt.colorbar(label = "Temperature in Celsius")
plt.show()

# %% Crop the tif to closer boundig boxes
from rasterio.mask import mask
xmin, ymin, xmax, ymax = config['bboxes']['munich_grid']
bbox = box(xmin, ymin, xmax, ymax)

# Crop the GeoTIFF to the bounding box
cropped, transform = mask(dataset=tif, shapes=[bbox], crop=True)

plt.imshow(cropped[0],'jet')
plt.colorbar(label = "Temperature in Celsius")
plt.show()

# %% Get value for weather station Oberhaching-Laufzorn
# Open the GeoTIFF file
with rasterio.open(path) as src:
    # Transform the coordinate to the pixel location
    row, col = src.index(11.5524, 48.0130)

    # Read the pixel value at the specified location
    value = src.read(1, window=((row, row+1), (col, col+1)))

# %% 
data = tif.read()[0]
# %%Create a 10x10 array filled with random numbers between 0 and 99
import random
data = np.array([[random.random() for _ in range(500)] for _ in range(500)])


# %% 
import numpy as np
#arr = np.array([[1, 2, 3],
#                [4, 5, 6],
#                [7, 8, 9]])


# def custom_filter(image):
#    return np.amax(image) - np.amin(image)

kernel = (3,3)
nPixel = kernel[0]*kernel[1]

from scipy.ndimage import generic_filter
result = generic_filter(cropped[0], np.mean, size=kernel, mode='constant', cval=15)


result = np.subtract(result,cropped[0]/nPixel)*(nPixel/(nPixel-1))

# 
difference = np.subtract(cropped[0],result)

# 
plt.imshow(cropped[0],'jet')
plt.colorbar(label = "Temperature in Celsius")

# Plot red dots for True values in the boolean array
y, x = np.where((difference > 1.5)) # &(difference < 4))
plt.scatter(x, y, color='green', marker='X', s=5)


plt.show()




# %%
plt.imshow(tif.read()[0],'jet',vmin=value)
plt.colorbar(label = "Temperature in Celsius")
plt.show()

# %%
#plt.show()

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



# %% Create an address searching fiel and select the respective grid
import geopandas
import geopy
import folium
from shapely.geometry import Point, Polygon
import pickle

# %%
def check_coordinates_in_bbox(latitude, longitude, bounding_box):
    point = Point(longitude, latitude)
    bbox_polygon = Polygon([(bounding_box[0], bounding_box[1]), (bounding_box[0], bounding_box[3]),
                            (bounding_box[2], bounding_box[3]), (bounding_box[2], bounding_box[1])])
    return bbox_polygon.contains(point)

# %% Load the grid
with open('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/uhi_model/grid/grid_250_a.pkl', 'rb') as file:
    grid = pickle.load(file)

# %% Search an adress

adress = input("Input your adress in the follwing format (Dorfstraße 5, Tübingen): ")

try:
    locator = geopy.geocoders.Nominatim(user_agent='myGeocoder')
    location = locator.geocode(adress + ', Germany')
except:
    print('Invalid adress')

if check_coordinates_in_bbox(location.latitude, location.longitude, bbox):
    is_inside = False
    i = 0

    while not is_inside and i < grid.shape[0]:
    
        polygon_coords = Polygon(grid.geometry[i])

        point_coordinates = (location.longitude, location.latitude)
        point = Point(point_coordinates)
    
        is_inside = polygon_coords.contains(point)

        i+=1

    if is_inside:
    
        polygon = Polygon(polygon_coords)

        # Calculate the centroid of the polygon for centering the map
        centroid = polygon.centroid

        # Create a folium map centered on the centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

        # Convert the polygon coordinates to a format compatible with folium
        polygon_coords = list(polygon.exterior.coords)
        polygon_coords = [[coord[1], coord[0]] for coord in polygon_coords]

        # Create a folium polygon and add it to the map
        folium.Polygon(locations=polygon_coords, color='blue', fill_color='blue', fill_opacity=0.4).add_to(m)

        # Add a marker for the point to the map
        folium.Marker(location=[location.latitude, location.longitude], popup="Point").add_to(m)

        # Display the map
        display(m)
    
    else:
        print('Adress doesnt fall into defined grid.')

else:
    print('Your adress doesnt fall into the defined area.')


# %% Create an mean tiff for all months
'''
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff/'

for period in periods:

    month = datetime.strptime(periods[0]['start'], '%Y-%m-%d %H:%M:%S').month

    minTemp, maxTemp = tempRange[month]
    
    dataOverview = dataQualityOverview([period], minTemp, 25, config)

    dataOverview = dataOverview[
        (pd.to_datetime(dataOverview['dateTime']).dt.hour >= 6) & 
        (pd.to_datetime(dataOverview['dateTime']).dt.hour <= 19) & 
        dataOverview.qualityFlag]
    
    # print(dataOverview)

    #for orbits in dataOverview['orbitNumber']:
    #    plot_by_key(orbits,'LSTE', config)
    #    plot_by_key(orbits,'Cloud', config)

    # print(dataOverview[dataOverview.qualityFlag].shape[0])

    # Store orbit numbers
    orbitNumbers = dataOverview['orbitNumber'] 

    name = path + period['start'].split(' ')[0] + '.tif'

    if dataOverview.shape[0] == 0: 
        continue
    
    elif dataOverview.shape[0] < 2:
        files = [
            f 
            for f in os.listdir(config['data']['ES_tiffs']) if orbitNumbers.iloc[0] in f
            ]
        # Extract path of lst and cloud
        lst=rasterio.open(
            os.path.join(config['data']['ES_tiffs'], [f for f in files if "LSTE" in f and f.endswith(".tif")][0])
            )
        cld=rasterio.open(
            os.path.join(config['data']['ES_tiffs'], [f for f in files if "Cloud" in f and f.endswith(".tif")][0])
            )

        # masked_array = np.ma.masked_array(lst.read()[0], mask=(cld.read()[0].astype(bool) | np.isnan(lst.read()[0])),fill_value=np.NaN)
        masked_array = np.where(cld.read()[0].astype(bool),np.NaN,lst.read()[0])

        tifffile.imwrite(name, masked_array)
    #    print('There are not enogh files to create a high quality mean tif!')
    #    continue
    else:
        # Create and store mean tiff
        meanTiff, maList = mergeTiffs(orbitNumbers, name, config)
'''


# %% TODO: The following could would be only necessary if we would use heatdays
# in addition to heatwaves
'''
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')

# %% Extract id for munich central
id = 3379
# Reduce the data to munich central
mCentral = dwd[dwd['STATIONS_ID'] == id].reset_index(drop=True)
# Convert the date column to datetime
mCentral['MESS_DATUM'] = mCentral['MESS_DATUM'].astype('datetime64[ns]')
# %%
mCentral_Summer = mCentral[(mCentral['MESS_DATUM'].dt.month >= 6) & (mCentral['MESS_DATUM'].dt.month <= 8)]

# %%
print(np.mean(mCentral_Summer.loc[mCentral_Summer.groupby(mCentral_Summer['MESS_DATUM'].dt.date)['TT_TU'].idxmin()].MESS_DATUM.dt.hour))

print(np.mean(mCentral_Summer.loc[mCentral_Summer.groupby(mCentral_Summer['MESS_DATUM'].dt.date)['TT_TU'].idxmax()].MESS_DATUM.dt.hour))

# %%

# %% Identify tropical days
tropicalDays = pd.to_datetime(mCentral[mCentral['TT_TU']>=30]['MESS_DATUM'].dt.date)
# %% Reduce to tropical days from last year
tropicalDays = set(tropicalDays[tropicalDays.dt.year > 2020])

# %%
from datetime import datetime, time, timedelta
tropicalPeriods = []

for dates in tropicalDays:
    tropicalPeriods.append(
        {'start': str(dates),
        'end': str(dates+ timedelta(days=1))
        })



# %%
mCentral['tropicalNight'] = False
# 
i = 0
while i < len(mCentral):
    
    if 6 < mCentral.iloc[i].MESS_DATUM.hour < 18:
        i+=1 
        continue

    else:
        temps=[]
        idxs=[]
        while 18 <= mCentral.iloc[i].MESS_DATUM.hour or mCentral.iloc[i].MESS_DATUM.hour <= 6:
            temps.append(mCentral.iloc[i].TT_TU)
            idxs.append(i)
            i+=1

            if i == len(mCentral):
                break
        
        if np.min(temps) >= 20:
            for index in idxs:
                mCentral.loc[index, 'tropicalNight'] = True

# %%
lastyear = mCentral[mCentral.MESS_DATUM.dt.year > 2020]

lastyear[lastyear['tropicalNight']]

# %%
tropicalPeriods.append({'start': '2021-06-18 18:00:00', 'end': '2021-06-19 06:00:00'})
tropicalPeriods.append({'start': '2021-06-19 18:00:00', 'end': '2021-06-20 06:00:00'})
tropicalPeriods.append({'start': '2022-06-19 18:00:00', 'end': '2022-06-20 06:00:00'})
tropicalPeriods.append({'start': '2022-07-13 18:00:00', 'end': '2022-07-14 06:00:00'})
# %%
import pickle
with open('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/tropicalPeriods.pkl', 'wb') as f:
    pickle.dump(tropicalPeriods, f)
'''

