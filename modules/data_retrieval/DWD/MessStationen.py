#%% Import packages
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests
import json
from random import randint
from time import sleep
import zipfile
import glob
import os
# %% Access to API
# r = requests.get('https://dwd.api.proxy.bund.dev/v30/stationOverviewExtended?stationIds=10865')
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
# Aceess url 
r = requests.get(url)
# Parse html
soup = BeautifulSoup(r.content, 'html5lib')
# Select all hrefs
hrefs = [a['href'] for a in soup.find_all('a', href=True)]
# %% Create final data frame
# Set time period
start_date = datetime.datetime(2018, 1, 1, 0)
end_date = datetime.datetime(2022, 12, 31, 23)
# Create date range
dates = pd.date_range(start_date, end_date, freq='H')
# Create empty data frame
stations_temp = pd.DataFrame({'MESS_DATUM': dates})
# %%
# Define relevant columns from stations data
cols = ['TT_TU', 'RF_TU']
#  Loop over all stations in and around Munich
for i in muenchen.Stations_id:
    # Select filename from hrefs
    filename = [h for h in hrefs if h.startswith('stundenwerte_TU_' + str(i).rjust(5,'0'))][0]
    # Unzip file from url + hrefs[0] and save it as txt file
    r = requests.get(url + filename)
    # Save zip file
    with open(filename, 'wb') as f:
        f.write(r.content)
    # Unzip file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(filename.replace('.zip',''))
    # Delete zip file
    os.remove(filename) 
    # Access txt file
    file = glob.glob(filename.replace('.zip','') + '/produkt_tu_stunde*')
    # Read txt file and save it as pandas dataframe
    station_data = pd.read_csv(file[0],
                               sep=';',
                               parse_dates = ['MESS_DATUM'],
                               date_format='%Y%m%d%H')
    # Rename columns
    station_data.rename({cols[0]:[c + '_' + str(station_data.STATIONS_ID.iloc[0]) for c in cols][0],
                         cols[1]:[c + '_' + str(station_data.STATIONS_ID.iloc[0]) for c in cols][1]},
                         axis=1,
                         inplace=True)
    # Merge data frame with station data
    stations_temp = (
        stations_temp.
        merge(station_data[['MESS_DATUM',
                            [c + '_' + str(station_data.STATIONS_ID.iloc[0]) for c in cols][0],
                            [c + '_' + str(station_data.STATIONS_ID.iloc[0]) for c in cols][1]]],
              how='left',
              on='MESS_DATUM')
              )
    # 
    sleep(randint(1,5))

# %% Replace all values below -100 in stations_temp with NaN
stations_temp[stations_temp.iloc[:, 1:] < -100] = np.nan
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

