#%% Import packages
import pandas as pd
import folium
import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests
import json
import zipfile
import glob
import os
import re
import shutil
from bs4 import BeautifulSoup
from folium.plugins import MarkerCluster
from random import randint
from time import sleep
# %% Set working directory
os.chdir('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/temp')
# %% Set url
url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/'
# %% Download TU_Stundenwerte_Beschreibung_Stationen.txt from url and save it as txt file
r = requests.get(url + 'TU_Stundenwerte_Beschreibung_Stationen.txt')
with open('TU_Stundenwerte_Beschreibung_Stationen.txt', 'wb') as f:
    f.write(r.content)
# %% Read txt file and save it as pandas dataframe
stations = pd.read_fwf('TU_Stundenwerte_Beschreibung_Stationen.txt',
                       skiprows=2,
                       encoding = "ISO-8859-1",
                       header=None,
                       names = ['Stations_id',
                                'von_datum',
                                'bis_datum',
                                'Stationshoehe',
                                'geoBreite',
                                'geoLaenge',
                                'Stationsname',
                                'Bundesland'])
# Transform date columns to datetime
stations['von_datum'] = pd.to_datetime(stations['von_datum'], format='%Y%m%d')
stations['bis_datum'] = pd.to_datetime(stations['bis_datum'], format='%Y%m%d')
# %% Select stations from relevant time period
stations = stations[stations['bis_datum'].dt.year >= 2022]
# # %% Plot stations on a map
m = folium.Map(location=[51.165691, 10.451526], zoom_start=6)
# %% Add marker cluster
marker_cluster = MarkerCluster().add_to(m)
# %% Add markers to map
for index, row in stations.iterrows():
    folium.Marker([row['geoBreite'], row['geoLaenge']], popup=row['Stationsname']).add_to(marker_cluster)
# %% Show map
m
# %% Select all stations that are in the relevant geoloical area
muenchen = stations[(stations['geoBreite'] > 47.99) & 
                    (stations['geoBreite'] < 48.405) &
                    (stations['geoLaenge'] > 11.18) &
                    (stations['geoLaenge'] < 12.0)]
# %% Select all href from url and save them as list
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
hrefs = [a['href'] for a in soup.find_all('a', href=True)]
# %% Create final data frame
# Set time period
start_date = datetime.datetime(2018, 1, 1, 0)
end_date = datetime.datetime(2022, 12, 31, 23)
# Create date range
dates = pd.date_range(start_date, end_date, freq='H')
# Create empty data frame
station_filter = pd.DataFrame({'MESS_DATUM': dates})
# %% filter scraped info
# Define relevant columns from stations data
cols = ['TT_TU', 'RF_TU']

station_temp = pd.DataFrame()

stations_id = muenchen.Stations_id.tolist()
stations_id = [str(s).rjust(5,'0') for s in stations_id]
j_stations_id = '|'.join(stations_id)
reg = 'stundenwerte_TU_' + j_stations_id + '_[0-9]{8}_20221231_hist.zip'
hrefs_final = list(filter(lambda x: re.findall(reg, x), hrefs))
# %% loop through chosen stations
for i in range(len(hrefs_final)):
    filename = hrefs_final[i]
    r = requests.get(url + filename, stream=True)
    # Save zip file
    with open(filename, 'wb') as f:
        f.write(r.content)
    # Unzip file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(filename.replace('.zip',''))
    # Access txt file
    file = glob.glob(filename.replace('.zip','') + '/produkt_tu_stunde*')
    # Read txt file and save it as pandas dataframe
    station_data = pd.read_csv(file[0], sep = ';')[["STATIONS_ID","MESS_DATUM","TT_TU","RF_TU"]]
    station_data['MESS_DATUM'] = pd.to_datetime(station_data['MESS_DATUM'], format='%Y%m%d%H')
    # Restrict to pre-specified time frame
    station_data = station_data.merge(station_filter, how="right", on='MESS_DATUM')
    # Append to data
    station_temp = pd.concat([station_temp, station_data])
    # Delete files
    shutil.rmtree(filename.replace('.zip',''))
    os.remove(filename)
    # Pause request
    sleep(randint(1,5))
# %% Data cleaning
station_temp[station_temp['RF_TU'] < 0]['RF_TU'] = np.nan
station_temp[station_temp['TT_TU'] < -50]['TT_TU'] = np.nan
#%% Create mask for August 2022
mask = (stations_temp['MESS_DATUM'].dt.year == 2022) & (stations_temp['MESS_DATUM'].dt.month == 8)
# %% Select data for August 2022
aug2022 = stations_temp[mask]
# %% Plot data for TT_TU_3379 and TT_TU_7431 in aug2022
plt.figure(figsize=(20,10))
plt.plot(aug2022['MESS_DATUM'], aug2022['TT_TU_3379'], label='TT_TU_3379')
plt.plot(aug2022['MESS_DATUM'], aug2022['TT_TU_7431'], label='TT_TU_7431')
plt.axhline(y=30, color='r', linestyle='--', label='30°C')
plt.legend(['Downtown', 'Outskirts', '30°C'])
plt.title('Temperature in Munich in August 2022')
plt.xlabel('Date')
plt.ylabel('Temperature in °C')
plt.show()