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

# Loop to invert heatwave
for dates in date_list:
    if not dateInHeatwave(datetime.datetime.strptime(dates['start'],'%Y-%m-%d %H:%M:%S'),hW_tD):
        inverted_HW.append(dates)
    elif not dateInHeatwave(datetime.datetime.strptime(dates['end'],'%Y-%m-%d %H:%M:%S'),hW_tD):
        inverted_HW.append(dates)

# %%
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')
# Extract id for munich central
id = 3379
# Reduce the data to munich central
mCentral = dwd[dwd['STATIONS_ID'] == id].reset_index(drop=True)
mCentral.drop('STATIONS_ID', inplace=True,axis=1)
# Convert the date column to datetime
mCentral['MESS_DATUM'] = mCentral['MESS_DATUM'].astype('datetime64[ns]')

# Filter the data for the year 2022
mCentral_2022 = mCentral[mCentral['MESS_DATUM'].dt.year == 2022]

minTemp = mCentral_2022.groupby(mCentral_2022['MESS_DATUM'].dt.month)['TT_TU'].min().to_list()
maxTemp = mCentral_2022.groupby(mCentral_2022['MESS_DATUM'].dt.month)['TT_TU'].max().to_list()

tempRange = {}

for i in range(1,13):
    diff = (maxTemp[i-1] - minTemp[i-1])/3
    tempRange[i] = [round(minTemp[i-1]-diff,2), round(maxTemp[i-1]+(diff*2),2)]
    
    
# %% Try cloud padding for the follwoing key '21476_003'
# Get lst tiff: 
key = '21476_003'

lstPath = [f for f in os.listdir(config['data']['ES_tiffs']) if f.endswith('.tif') and 'LST' in f and key in f]
cloudPath = [f for f in os.listdir(config['data']['ES_tiffs']) if f.endswith('.tif') and 'CLOUD' in f and key in f]

lst=rasterio.open(os.path.join(config['data']['ES_tiffs'], lstPath[0]))
cld=rasterio.open(os.path.join(config['data']['ES_tiffs'], cloudPath[0]))

img_lst = lst.read()[0]
img_cld = cld.read()[0]

masked_array = np.ma.masked_array(img_lst, mask=(img_cld.astype(bool) | (lst.read()[0]<1)))

plt.imshow(masked_array,'jet')
plt.colorbar(label = "Temperature in Celsius")
plt.show()

# ChatGPT suggestion
'''
import numpy as np
from scipy.ndimage import binary_dilation

def add_pixels_around_mask(mask, radius):
    dilated_mask = binary_dilation(mask, iterations=radius)
    return dilated_mask

# Example usage
# Assuming you have a binary mask 'cloud_mask' and you want to add a radius of 3 pixels
cloud_mask = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])

radius = 3
dilated_mask = add_pixels_around_mask(cloud_mask, radius)

print("Original mask:")
print(cloud_mask)

print("Dilated mask:")
print(dilated_mask)

'''


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

# %% Count the number of files for a specific period
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
summer = [{'start': '2022-01-01 00:00:00', 'end': '2022-01-03 00:00:00'}]

confirmation = input("Do you want to create tiffs for hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    processHF(summer, tempRange, config)
else:
    print("Function execution cancelled.")

# %% Check if each LSTE.h5 has a respective tif
raw_files = listdir(config['data']['ES_raw'])

# Extract keys
keys = set(
    [
    files.split('_')[3] + '_' + files.split('_')[4]
    for files in raw_files if 'GEO' in files
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



# %% TODO: Create a loop that creates and stores a mean tif for each first and second half of a month
from datetime import datetime, timedelta

def split_period(period):
    result = []
    format_str = '%Y-%m-%d %H:%M:%S'
    
    start_date = datetime.strptime(period[0]['start'], format_str)
    end_date = datetime.strptime(period[0]['end'], format_str)
    current_date = start_date
    
    while current_date < end_date:
        month_start = current_date.replace(day=1)
        next_month_start = (month_start + timedelta(days=31)).replace(day=1)
        half_duration = (next_month_start - month_start) // 2
        first_half_end = month_start + half_duration
        second_half_start = first_half_end + timedelta(seconds=1)
        second_half_end = next_month_start - timedelta(seconds=1)
        
        result.append({
            'start': current_date.strftime(format_str),
            'end': first_half_end.strftime(format_str)
        })
        
        result.append({
            'start': second_half_start.strftime(format_str),
            'end': second_half_end.strftime(format_str)
        })
        
        current_date = next_month_start
    
    return result

period = [{'start': '2022-01-01 00:00:00', 'end': '2023-01-01 00:00:00'}]
periods = split_period(period)




# %% Create an overview over all existing tiffs
# TODO: Before serializing the code, change the quality measures
# Try to be more sensitive and set a measure for nearly quadratic pictures
# TODO: Before serializing, check if there is enough data to split by
# day and night

#dataOverview = dataQualityOverview([periods[7]], config)
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/meanTiff/'

flags = ['day', 'night']

# Delete existing files
files = os.listdir(path)

for f in files: 
    os.remove(os.path.join(path,f))

# %%

dataOverview = dataQualityOverview([periods[7]], 0 ,config)

# %%
plot_by_key('21476_003', config)

# %%
temps = [2.1, 4.8]

for period in periods:
    
    dataOverview = dataQualityOverview([period],0 ,config)

    #flag = 'day'

    dataOverview = dataOverview[
        (pd.to_datetime(dataOverview['dateTime']).dt.hour >= 4) & 
        (pd.to_datetime(dataOverview['dateTime']).dt.hour <= 8) & 
        dataOverview.qualityFlag]
    
    print(dataOverview[dataOverview.qualityFlag].shape[0])


# %%
    # for flag in flags:

        #if flag == 'day': 
            
         #   dataOverview = dataOverview[
         #       (pd.to_datetime(dataOverview['dateTime']).dt.hour >= 6) & 
         #       (pd.to_datetime(dataOverview['dateTime']).dt.hour <= 18) & 
         #       dataOverview.qualityFlag]

       # elif flag == 'night':
       #     dataOverview = dataOverview[
       #         (pd.to_datetime(dataOverview['dateTime']).dt.hour >= 18) & 
       #         (pd.to_datetime(dataOverview['dateTime']).dt.hour <= 6) & 
       #         dataOverview.qualityFlag]

    if dataOverview.shape[0] < 3:
        print('There are not enogh files to create a high quality mean tif!')
        continue
    
    name = path + period['start'].split(' ')[0] + flag + '.tif'
    
    # Store orbit numbers
    orbitNumbers = dataOverview['orbitNumber'] 
    # Create and store mean tiff
    meanTiff, maList = mergeTiffs(orbitNumbers, name, config)

# %% Delete existing tiffs

for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)
    
    # Check if the path is a file (not a subdirectory)
    if os.path.isfile(file_path):
        # Delete the file
        os.remove(file_path)


# %% Plot all tifs in the directory

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







# %% Create a Dataframe to check the quality of all relevant tiffs (in heatwave)
# Relevant input is hW_tD and inverted_HW 
dataQ = dataQualityOverview(inverted_HW, 10, config)

# %% Plot LST tiff by key
plot_by_key('19797_003',config)

# %%
f_lst = h5py.File('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/raw_h5/ECOSTRESS_L2_LSTE_25115_005_20221210T094045_0601_01.h5')


# %% 
# Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 12 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 16) & 
    dataQ['qualityFlag']
    ]

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
morning = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 5) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 8) & 
    dataQ['qualityFlag']
    ]

# Set name
name = 'mean_Morning_nonHT.tif'

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
# TODO: Add water to the map
morning_map.save('morning_nonHT.html')

# %% 
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/'
morningHW = rasterio.open(path + 'mean_Morning_HT.tif')
afternoonHW = rasterio.open(path +'mean_afterNoon_HT.tif')

# %%

diff_NoonMorning = afternoonHW.read()[0] - morningHW.read()[0]

# %%
diff_NoonMorning[diff_NoonMorning < 0 ] = np.NaN

# %%
# vmin=-2.5, vmax=19
plt.imshow(diff_NoonMorning,'jet_r', vmin=0, vmax=19)
plt.colorbar(label = "Temperature in Celsius")

plt.show()



# %% Tropical day, tropical night

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

