#%% start-up
from preprocessing_DWD import DWDScraper

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import yaml
config_path = '/home/tu/tu_tu/tu_zxmny46/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# %%
S = DWDScraper()
#%% get all stations
S.get_all_stations("2020-01-01", "2022-12-31")
#%% get all stations
s = pd.read_csv(config['data']['dwd']+'/meta/stations.csv')
s.head()
# %%
ids = S.get_relevant_station_ids(s)
# %%
s[s.STATIONS_ID.isin(ids)]
# %%
S.scrape("2020-01-01","2022-12-31",ids)
# %%
dwd = pd.read_csv(config['data']['dwd']+'/dwd.csv')
dwd['MESS_DATUM'] = pd.to_datetime(dwd['MESS_DATUM'], format='%Y-%m-%d %H')
dwd.head()
# %%
dwd = dwd.merge(s[["STATIONS_ID","GEOBREITE","GEOLAENGE","STATIONSNAME"]], how="inner", on="STATIONS_ID")
dwd.head()
# %%
dwd.STATIONSNAME.value_counts(dropna=False)
# %%
s['BIS_DATUM'] = pd.to_datetime(s['BIS_DATUM'], format='%Y-%m-%d')
stations = s[s['BIS_DATUM'].dt.year >= 2022]
m = folium.Map(location=[51.165691, 10.451526], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)
for index, row in stations.iterrows():
    folium.Marker([row['GEOBREITE'], row['GEOLAENGE']], popup=row['STATIONSNAME']).add_to(marker_cluster)
# %% Show map
m
# %%
mask = (dwd['MESS_DATUM'].dt.year == 2022) & (dwd['MESS_DATUM'].dt.month == 8) & (dwd.STATIONS_ID.isin([3379,7431]))
aug2022 = dwd[mask]
# %% Plot data for TT_TU_3379 and TT_TU_7431 in aug2022
plt.figure(figsize=(20,10))
sns.lineplot(data=aug2022, x='MESS_DATUM', y='TT_TU', hue='STATIONSNAME')
plt.axhline(y=30, color='r', linestyle='--', label='30°C')
plt.legend(['Downtown', 'Outskirts', '30°C'])
plt.title('Temperature in Munich in August 2022')
plt.xlabel('Date')
plt.ylabel('Temperature in °C')
plt.show()