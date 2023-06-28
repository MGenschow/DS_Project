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


# %% Create header 
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


# %% Import heatwaves
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')

# Combine dates to periods format of heatwaves TODO: Create a function
heatwaves = heatwave_transform(dates)
# Add heatwaves from 2021
heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-22 00:00:00'})
heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-16 00:00:00'})

# Import tropical timeperiods
tropicalDays = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/tropicalPeriods.pkl')


# Combine heatwaves and heatdays
hW_tD = heatwaves + tropicalDays

#  Invert the heatwave
inverted_HW = invertHeatwave(hW_tD)


# Calculate the temperature range for each month
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')

tempRange = calculateTempRange(dwd)
    

# %% Set spatial filter;
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


# %% QC: Count the number of files for a specific period
month = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]
types = ['GEO','CLOUD', 'LSTE']
# Loop over files
for t in types: 
    files =  [
        f
        for f in listdir(config['data']['ES_raw']) 
        if t in f and dateInHeatwave(datetime.datetime.strptime(f.split('_')[5], '%Y%m%dT%H%M%S'), month)
        ]
        
    print(f'There are {len(files)} {t} files')


# %% Create a tiff for each unique scene
# Define period to create tiffs
period = [{'start': '2022-01-01 00:00:00', 'end': '2022-01-15 00:00:00'}]

confirmation = input("Do you want to create tiffs for hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    processHF(period, tempRange, config)
else:
    print("Function execution cancelled.")


# %% QC: Check if each LSTE.h5 has a respective tif
raw_files = listdir(config['data']['ES_raw'])

# Extract keys
keys = set(
    [
    files.split('_')[3] + '_' + files.split('_')[4]
    for files in raw_files if 'LSTE' in files
    ]
)

tiffs = [
    f 
    for f in listdir(config['data']['ES_tiffs'])
    if isfile(join(config['data']['ES_tiffs'], f)) and
    f.endswith('.tif')
    ]

for k in keys:
    l = len([f for f in tiffs if k in f])

    if l < 1:
        print(f'There are no tiff files for the key {k}')
        print([f for f in raw_files if k in f])


# %% Create a loop that creates and stores a mean tif for each first and second half of a month

# Split period
period = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]
periods = split_period(period, split_half=False)


# %% Create an overview over all existing tiffs

path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff/'


for period in [periods[5]]:

    month = datetime.strptime(periods[0]['start'], '%Y-%m-%d %H:%M:%S').month

    minTemp, maxTemp = tempRange[month]
    
    dataOverview = dataQualityOverview([period], minTemp, 45, config)

    dataOverview = dataOverview[
    #    (pd.to_datetime(dataOverview['dateTime']).dt.hour >= 5) & 
    #    (pd.to_datetime(dataOverview['dateTime']).dt.hour <= 20) & 
        dataOverview.qualityFlag]
    
    print(dataOverview)

    #for orbits in dataOverview['orbitNumber']:
    #    plot_by_key(orbits,'LSTE', config)
    #    plot_by_key(orbits,'Cloud', config)

    # print(dataOverview[dataOverview.qualityFlag].shape[0])

    #if dataOverview.shape[0] < 2:
    #    print('There are not enogh files to create a high quality mean tif!')
    #    continue
    
    name = path + period['start'].split(' ')[0] + '.tif'
    
    # Store orbit numbers
    orbitNumbers = dataOverview['orbitNumber'] 
    # Create and store mean tiff
    meanTiff, maList = mergeTiffs(orbitNumbers, name, config)

# %% 

tif = rasterio.open('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff/2022-06-01.tif')
plt.imshow(tif.read()[0],'jet')
plt.colorbar(label = "Temperature in Celsius")

plt.show()

# %% Plot all mean tiffs
# Set the directory path where the TIFF images are located
directory = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff'

# Get a list of all TIFF files in the directory
tiff_files = [file for file in os.listdir(directory) if file.endswith('.tiff') or file.endswith('.tif')]
tiff_files.sort()


# Loop through each TIFF file and plot it
for tiff_file in tiff_files:
    # Create the full file path
    file_path = os.path.join(directory, tiff_file)
    
    tif = rasterio.open(file_path)
    plt.imshow(tif.read()[0],'jet')
    plt.colorbar(label = "Temperature in Celsius")

    plt.show()


# %% 
meanTiff(hW_tD,insideHeatwave=True, config=config)
meanTiff(inverted_HW, insideHeatwave=False, config=config)

# %%
# Plot tiff
tif = rasterio.open('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/avgMorning_nonHW.tif')

plt.imshow(tif.read()[0],'jet') # ,vmin=16, vmax=45)
plt.colorbar(label = "Temperature in Celsius")

plt.show()



# %% 
dataQ = dataQualityOverview(inverted_HW, 0, 35, config)

# %% Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 12 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 16) & 
    dataQ['qualityFlag']
    ]
# %%
for o in afterNoon.orbitNumber:
    plot_by_key(o,'LSTE', config)
    plot_by_key(o,'Cloud', config)
# %%
name = 'mean_afterNoon_nonHT.tif'

# Store orbit numbers
orbitNumbers = afterNoon['orbitNumber']
# Create and store mean tiff
meanAfterNoon, maList = mergeTiffs(orbitNumbers, name, config)

path = config['data']['ES_tiffs'].replace('geoTiff/','') + name

# HW ranges from 20.2 to 44.8
# Inverted HW ranges from 16.8 to 38

# Plot tiff
tif = rasterio.open(path)

plt.imshow(tif.read()[0],'jet',vmin=16, vmax=45)
plt.colorbar(label = "Temperature in Celsius")

plt.show()


# %%Plot arrays
arrays_subplot(maList)

# %% Plot tif over a interactive open street map
map_afternoon = tiffs_to_foliumMap(path)
map_afternoon

# Save after noon 
map_afternoon.save('afterNoon_nonHT.html')



# %% # Create DF with relevant tifs for the morning
dataQ = dataQualityOverview(inverted_HW, 0, 25, config)

morning = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 5) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 8) & 
    dataQ['qualityFlag']
    ]

# %% 
for o in morning[morning.qualityFlag].orbitNumber:
    plot_by_key(o,'LSTE', config)
    plot_by_key(o,'Cloud', config)

# %%
# Set name
name = mean_Morning_nonHT.tif'

# Store orbit numbers
orbitNumbers = morning['orbitNumber']

# Create and store mean tiff
meanMorning, maList = mergeTiffs(orbitNumbers, name, config)

# Set path 
path = config['data']['ES_tiffs'].replace('geoTiff/','') + name

# %% Plot tiff

# HW ranges from 16.6 to 35.5
# Inverted HW ranges from 9.6 to 24.2

tif = rasterio.open(path)
plt.imshow(tif.read()[0],'jet', vmin=9, vmax=36)
plt.colorbar(label = "Temperature in Celsius")

plt.show()

# %% Plot arrays
arrays_subplot(maList)

# %% Plot tif over a interactive open street map
morning_map = tiffs_to_foliumMap(path)
morning_map

# Save folium map
# TODO: Reduce map to munich 

morning_map.save('morning_nonHT.html')

# %% 
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/'
morningHW = rasterio.open(path + 'mean_Morning_HT.tif')
afternoonHW = rasterio.open(path +'mean_afterNoon_HT.tif')


# %%
diff_NoonMorning = afternoonHW.read()[0] - morningHW.read()[0]
diff_NoonMorning[diff_NoonMorning < 0 ] = np.NaN

# %%
# vmin=-2.5, vmax=19
plt.imshow(diff_NoonMorning,'jet_r', vmin=0, vmax=19)
plt.colorbar(label = "Temperature in Celsius")

plt.show()






# %% TODO: Add to DWD data: Tropical day, tropical night

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

