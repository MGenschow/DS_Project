# %%
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import preprocessing_orthophotos

from shapely.geometry import box, Polygon, mapping
import pandas as pd

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config



# %%
# Define bbox for the area of interest
bbox = [11.55, 48.16, 11.56, 48.17]
bbox = box(*bbox)

loader = preprocessing_orthophotos.OrthophotoLoader(config['data']['orthophotos'], bbox, download=True)
relevant_tiles = loader.get_relevant_tiles()

# %%
from shapely.geometry import mapping
features = []
for polygon in list(relevant_tiles['polygon_4326']):
    features.append({
                'type': 'Feature',
                'geometry': mapping(polygon),
                'properties': {}
            })
geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

style1 = {"color": "#1052EA","weight": 1,"opacity": 0.65}
style2 = {'color': '#DE1D0D'}

import folium
m = folium.Map(location=[bbox.centroid.y, bbox.centroid.x], zoom_start=13)
folium.GeoJson(geojson, style_function=lambda x:style1).add_to(m)
folium.GeoJson(bbox.__geo_interface__, style_function=lambda x:style2).add_to(m)

m

# %%
loader.print_report()

# %%
loader.download_missing_tiles()





# %%
