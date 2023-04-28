# %%
import pandas as pd 
import requests
import urllib
import os
from shapely.geometry import box, mapping, Polygon
import rasterio.warp
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import pickle

# %%
import yaml
config_path = '/Users/maltegenschow/Documents/Uni/SoSe23/Data Science Project/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config

# %%
bbox = box(*[11.55, 48.16, 11.56, 48.17])

# %% [markdown]
# ###  Read Data 
data = pickle.load(open(os.path.join(config['data']['building_boxes'],'processed_building_boxes', 'building_boxes.pkl'), 'rb'))
data

# %% [markdown]
# ### Plot Building Boxes on Map

# %%
features = []
for polygon in data['geometry_4326']: 
    features.append({
            'type': 'Feature',
            'geometry': mapping(polygon),
            'properties': {}
        })


geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

# %%
import folium
m = folium.Map(location=[bbox.centroid.y, bbox.centroid.x], zoom_start=15)
folium.TileLayer('openstreetmap').add_to(m)
folium.GeoJson(bbox.__geo_interface__, style_function=lambda x:{'color':'#DE1D0D'}).add_to(m)
folium.GeoJson(geojson).add_to(m)

# %% [markdown]
# ### Add Image over map

# %%
import rioxarray
import numpy as np

dst = rioxarray.open_rasterio('/Users/maltegenschow/Documents/Uni/SoSe23/Data Science Project/data/orthophotos/raw_tiles/32689_5337.tif')
dst = dst.rio.reproject('EPSG:4326')
img = np.dstack((dst.values[0], dst.values[1], dst.values[2]))

image_bounds = box(*dst.rio.bounds())

# %%
import folium
folium.GeoJson(image_bounds.__geo_interface__).add_to(m)
min_x, min_y, max_x, max_y = dst.rio.bounds()

corner_coordinates = [[min_y, min_x], [max_y, max_x]]
folium.raster_layers.ImageOverlay(img, bounds=corner_coordinates,opacity=0.9).add_to(m)

m

# %% [markdown]
# ### Test Subsetting of image to one rooftop

# %%
data['area'] = data['geometry_4326'].area
data.nlargest(3, 'area')

# %%
roof = data.nlargest(1, 'area')['geometry'].iloc[0]
roof

# %%
roof.wkt

# %%
with rasterio.open('/Users/maltegenschow/Documents/Uni/SoSe23/Data Science Project/data/orthophotos/raw_tiles/32689_5337.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, [roof], crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("masked_roof.tif", "w", **out_meta) as dest:
    dest.write(out_image)

# %%
roof_picture = rioxarray.open_rasterio('masked_roof.tif')
img = np.dstack((roof_picture.values[0], roof_picture.values[1], roof_picture.values[2]))

# %%
plt.imshow(img)



# %%
