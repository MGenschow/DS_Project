# %%
import pandas as pd
import xml.etree.ElementTree as ET 
import urllib
import re
import os
import geopandas as gpd
import rasterio
import rasterio.warp
from shapely.geometry import Polygon
from shapely.geometry import mapping

# Make sure imports from modules in parent directory will work
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import preprocessing_orthophotos

import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# ### Download and Parse all metalink files to create complete index of DOP40 orthophotos

# %%
indexer = preprocessing_orthophotos.OrthophotoIndexer()
#indexer.download_metalink_per_regierungsbezirk(override=False)
indexer.create_complete_index()

# %%
df = pd.read_pickle(os.path.join(config['data']['orthophotos'],'metalink_files','complete_index.pkl'))
df.head(4)

# %% [markdown]
# ### Check for duplicates

# %%
# Check duplicates due to double mentioning in different Regierungsbezirken
df.duplicated(subset = df.columns.to_list()[:-1]).sum()

# %%
# Drop duplicates
df.drop_duplicates(subset = df.columns.to_list()[:-1], keep = 'first', inplace = True)

# %% [markdown]
# ### Show all available orthophotos: 

# %%
colors = {'Oberbayern': '#ff0000',
          'Niederbayern': '#00ff00',
          'Oberpfalz': '#0000ff',
          'Oberfranken': '#ffff00',
          'Mittelfranken': '#ff00ff',
          'Unterfranken': '#00ffff',
          'Schwaben': '#000000'}

def create_geojson_by_regierungsbezirk(df, bezirk):
    polygon_list = df[df.Regierungsbezirk == bezirk].polygon_4326.to_list()
    features = []
    for i, polygon in enumerate(polygon_list):
        # create a dictionary of properties for each polygon
        properties = {
            'name': bezirk,
            'color': colors[bezirk],
        }
        features.append({
            'type': 'Feature',
            'geometry': mapping(polygon),
            'properties': properties
        })

    # create a GeoJSON object with all the features
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    return geojson



# %%
geojson_files = {bezirk:create_geojson_by_regierungsbezirk(df, bezirk) for bezirk in df.Regierungsbezirk.unique()}
geojson_files.keys()

import folium
m = folium.Map(location=[48.9, 11.4], zoom_start=7)
# Add the GeoJSON as a layer to the map
for geojson in geojson_files.values():
    folium.GeoJson(geojson,
                style_function=lambda feature: {
                    'fillColor': feature['properties']['color'],
                    'color': feature['properties']['color'],
                    'weight': 1,
                    'fillOpacity': 0.5,
                    'tooltip': feature['properties']['name']
                }).add_to(m)

m

# %%



