#%% preparation
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import folium
import folium.raster_layers
import folium.features
import rasterio
import rioxarray as rxr

from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon
from shapely.geometry import box
from branca.colormap import LinearColormap

home_directory = os.path.expanduser( '~' )
os.chdir(home_directory + '/DS_Project/modules')

import warnings
warnings.filterwarnings("ignore")
#%% import other scripts
from disaggregate import *
#%% fix hyperparameters
grid_size_meters = 100
# coordinates = config['bboxes']['munich']
coordinates = config['bboxes']['munich_grid']
# coordinates = [11.547582, 48.114226, 11.627263, 48.155554]
#%% create grids
polygon_gdf = create_polygon_from_coord(coordinates=coordinates)
grid = divide_polygon_into_grid(polygon_gdf.geometry[0], grid_size_meters)
print("Number of grid elements: " + str(len(grid)))
with open(path_grid + 'grid_' + str(grid_size_meters) + '_b.pkl', 'wb') as file:
    pickle.dump(grid, file)
#%% get raw input data
with open(path_raw + 'input.pkl', 'rb') as file:
    inp = pickle.load(file)
with open(config['data']['city_3d_model'] + '/processed/processed_roofs.pkl', 'rb') as file:
    wind = pickle.load(file)
inp.to_crs(crs='EPSG:4326', inplace=True)
bbox = box(*coordinates)
inp = inp[inp.geometry.intersects(bbox)]
wind = wind[wind.geometry_4326.intersects(bbox)]
#%% get surface fractions
surface_labels = inp.label.unique().tolist()
result = calculate_surface_coverage(grid, inp, surface_labels)
surface_df = convert_dict_to_cols(result)
#%% get average height
height = calculate_average_height(grid, wind)
features = gpd.GeoDataFrame(pd.merge(surface_df, height, on='id', how='inner'))
print("Number of rows in feature dataframe: " + str(len(features)))
#%% get raw lst data
path_tif = config['data']['data'] + '/ECOSTRESS/avgAfterNoon_HW.tif'
lst_array = rxr.open_rasterio(path_tif)
#%% extract lst data for grid
grid['nLST'] = grid.apply(lambda row: naive_pixel_mean_wrapper(row, lst_array), axis=1)
grid['wLST'] = grid.apply(lambda row: weighted_pixel_mean_wrapper(row, lst_array), axis=1)
#%% join data
final = gpd.GeoDataFrame(pd.merge(grid, features, on='id', how='inner'))
print("Number of rows in final dataframe: " + str(len(final)))
print(final.head(10))
#%% save data
with open(path + 'final_' + str(grid_size_meters) + '_b.pkl', 'wb') as file:
    pickle.dump(final, file)