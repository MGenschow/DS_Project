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
heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-22 00:00:00'})
heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-16 00:00:00'})

# Import tropical timeperiods
tropicalDays = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/tropicalPeriods.pkl')

# Combine both
hW_tD = heatwaves + tropicalDays

# %%
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

print(len(date_list))

# NOTE: Removing the element makes the loop jumping over the element after that 
for dates in date_list:
    if dateInHeatwave(datetime.datetime.strptime(dates['start'],'%Y-%m-%d %H:%M:%S'),[hW_tD[0]]) or dateInHeatwave(datetime.datetime.strptime(dates['end'],'%Y-%m-%d %H:%M:%S'),[hW_tD[0]]):
        # date_list.remove(dates)
        print(dates)
    elif dateInHeatwave(datetime.datetime.strptime(dates['end'],'%Y-%m-%d %H:%M:%S'),[hW_tD[0]]):
        # date_list.remove(dates)
        print(dates)

print(len(date_list))


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
#month = [{'start': '2022-08-01 00:00:00', 'end': '2022-09-01 00:00:00'}]
# %%
# NOTE: Can take, depending on the parameters, quite some time 
# (up to several hours)
confirmation = input("Do you want to download the hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    #for temporalFilter in heatwaves:
    # for temporalFilter in tropicalDays:
    for temporalFilter in month:
        downloadH5(credentials, headers, temporalFilter, spatialFilter, config)
else:
    print("Loop execution cancelled.")


# %% Count number of files
types = ['GEO','CLOUD', 'LSTE']
# Loop over files
for t in types: 
    files =  [
        f
        for f in listdir(config['data']['ES_raw']) 
        if t in f and dateInHeatwave(datetime.datetime.strptime(f.split('_')[5], '%Y%m%dT%H%M%S'), hW_tD)]
        
    print(f'There are {len(files)} {t} files')

# %% Show missing files
onlyfiles = [
    f 
    for f in listdir(config['data']['ES_raw']) 
    if isfile(join(config['data']['ES_raw'], f))
    ]
    
# Extract keys
keys = [
    files.split('_')[3] + '_' + files.split('_')[4] 
    for files in onlyfiles 
    # if 'LSTE' in files and  dateInHeatwave(datetime.datetime.strptime(files.split('_')[5], '%Y%m%dT%H%M%S'), hW_tD)
    ]
# Reduce to unique
unique_keys = set(keys)

for key in unique_keys:
    # Check if all files are aivalable
    if len([f for f in onlyfiles if key in f]) != 3:
        files = [f for f in onlyfiles if key in f]
        print([f for f in onlyfiles if key in f])
        print(datetime.datetime.strptime(files[0].split('_')[5], '%Y%m%dT%H%M%S'))

# %% Create a tiff for each unique scene
# processHF(heatwaves, config)
processHF(hW_tD, config)

# %% Plot cipped tiffs 
'''
onlyfiles = [
    f 
    for f in listdir(config['data']['ES_tiffs']) 
    if isfile(join(config['data']['ES_tiffs'], f))
    ]
keys = [
    files.split('_')[3] + '_' + files.split('_')[4] 
    for files in onlyfiles
    ]
unique_keys = set(keys)
# 
path = config['data']['ES_tiffs']
# %%
for key in unique_keys:
    # Check if all files
    lst=rasterio.open(
        os.path.join(path, [f for f in onlyfiles if key in f and "LSTE" in f and 'QC' not in f and f.endswith(".tif")][0])
        )
    cld=rasterio.open(
        os.path.join(path, [f for f in onlyfiles if key in f and "Cloud" in f and f.endswith(".tif")][0])
        )
    qc=rasterio.open(
        os.path.join(path, [f for f in onlyfiles if key in f and 'QC' in f and f.endswith(".tif")][0])
        )
    # Plot the images
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the LSTE image
    axs[0].imshow(lst.read(1), cmap='jet')
    axs[0].set_title('LSTE')

    # Plot the Cloud image
    axs[1].imshow(cld.read(1), cmap='gray')
    axs[1].set_title('Cloud')

    # Plot the QC image
    axs[2].imshow(qc.read(1), cmap='jet')
    axs[2].set_title('QC')

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the plot
    plt.show()

# %%
for key in unique_keys:
    qc=rasterio.open(
        os.path.join(path, [f for f in onlyfiles if key in f and 'QC' in f and f.endswith(".tif")][0])
        )
    unique, counts = np.unique(qc.read(), return_counts=True)
    print(np.asarray((unique, counts)).T)
    print(' ')
'''
# %% Create a Dataframe to check the quality of all relevant tiffs (in heatwave)
dataQ = dataQualityOverview(hW_tD, config)

# %% Plot LST tiff by key
key = '22409_006'

onlyfiles = [
    f 
    for f in listdir(config['data']['ES_tiffs']) 
    if isfile(join(config['data']['ES_tiffs'], f)) and f.endswith('.tif')
    ]

lst = rioxarray.open_rasterio(
    config['data']['ES_tiffs'] + 
    [f for 
    f in [p for p in onlyfiles if key in p] 
    if 'LSTE' in f and '.tif' in f][0]
    )
img = np.array(lst)[0]
plt.imshow(img, cmap='jet')
plt.show()

# %% 
# Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 12 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 16) & 
    dataQ['qualityFlag']
    ]

name = 'mean_afterNoon.tif'

# Store orbit numbers
orbitNumbers = afterNoon['orbitNumber'] 
# Create and store mean tiff
meanAfterNoon, maList = mergeTiffs(orbitNumbers, name, config)

path = config['data']['ES_tiffs'].replace('geoTiff/','') + name

# %% Plot tiff
tif = rasterio.open(path)

plt.imshow(tif.read()[0],'jet')
plt.colorbar(label = "Temperature in Celsius")

plt.show()

# %%Plot arrays
arrays_subplot(maList)

# %% Plot tif over a interactive open street map
map_afternoon = tiffs_to_foliumMap(path)
map_afternoon

# Save after noon 
#map_afternoon.save('afterNoon.html')


# %% # Create DF with relevant tifs for the morning
morning = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 5) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 8) & 
    dataQ['qualityFlag']
    ]

# Set name
name = 'mean_Morning.tif'

# Store orbit numbers
orbitNumbers = morning['orbitNumber']

# Create and store mean tiff
meanMorning, maList = mergeTiffs(orbitNumbers, name, config)

# Set path 
path = config['data']['ES_tiffs'].replace('geoTiff/','') + name

# %% Plot tiff
tif = rasterio.open(path)
plt.imshow(tif.read()[0],'jet')

# %% Plot arrays
arrays_subplot(maList)

# %% Plot tif over a interactive open street map
morning_map = tiffs_to_foliumMap(path)
morning_map

# Save folium map
morning_map.save('morning.html')

# %%
# TODO: Reduce map to munich 
# TODO: Add water to the map

# %%
'''
Lst_files = [
    f for f in listdir(config['data']['ES_raw'])
    if 'LST' in f
]
# %%
f_lst = h5py.File(os.path.join(config['data']['ES_raw'], Lst_files[0]))

# %%
f_lst['SDS'].keys()

data = np.array(f_lst['SDS']['QC'])

plt.imshow(data,'jet')

# %%
f_lst = h5py.File(os.path.join(config['data']['ES_raw'], Lst_files[0]))

# Store relative paths of elements in list
eco_objs = []
f_lst.visit(eco_objs.append)

# Show datasets in f_lst
lst_SDS = [str(obj) for obj in eco_objs if isinstance(f_lst[obj], h5py.Dataset)]

# Store name of relevant dataset
sds = ['LST','QC']
# Extract relevant datasets
lst_SDS  = [dataset for dataset in lst_SDS if dataset.endswith(tuple(sds))]

# %%
# Read in data
lst_SD = f_lst[lst_SDS[0]][()]
# %%
qc_SD = f_lst[lst_SDS[1]][()]

# %%
def get_bit(x):
    return int('{0:016b}'.format(x)[0:2])

# Vectorize function
get_zero_vec = np.vectorize(get_bit)
# %%

qc_SDS = get_zero_vec(qc_SD)
# %%

class_labels = ['poor', 'marginal', 'good', 'excellent']
class_colors = ['red', 'green', 'blue', 'orange']

# Plot the data
plt.imshow(qc_SDS)
# %%
# Create a custom legend with class labels and colors
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
    for label, color in zip(class_labels, class_colors)
    ]
# Add the legend to the plot
plt.legend(handles=legend_elements)

# Show the plot
plt.show()


# %%
'''
# %% Tropical day, tropical night
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
# %%

