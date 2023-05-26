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

# Import all functions from utils
from utils import *

# %% Load login credentials and the config file 
# Import credentials from credentials.yml file; TODO adapat
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

# %% Import heatwaves; TODO: Understand definition of heatwaves
# TODO: Is it possible to soften the definition?
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')

# Combine dates to periods format of heatwaves
heatwaves = heatwave_transform(dates)


# %% Set spatial filter;
spatialFilter =  {
    'filterType' : "mbr",
    'lowerLeft' : {
        'latitude' : config["bboxes"]["munich"][1],
        'longitude' : config["bboxes"]["munich"][0]
        },
        'upperRight' : {
            'latitude' : config["bboxes"]["munich"][3],
            'longitude' : config["bboxes"]["munich"][2]}
            }
            
# %% Download all files corresponding to the heatwaves
# NOTE: Can take, depending on the parameters, quite some time 
# (up to several hours)

# for temporalFilter in heatwaves:
#    downloadH5(credentials, headers, temporalFilter, spatialFilter, config)

# %% Extract all unique keys
# Get all filepaths
onlyfiles = [f for f in listdir(config['data']['ES_raw']) if isfile(join(config['data']['ES_raw'], f))]

# Pull put unique keys
unique_keys = set([files.split('_')[3] + '_' + files.split('_')[4] for files in onlyfiles])

# %% 
# Create tif for all files corresponding to the heatwaves
path = config['data']['ES_raw']

# Loop over all unique keys in the raw_h5 folder
for key in unique_keys:
    
    # Get all file paths containing the key
    lstF = [f for f in onlyfiles if key in f and 'LSTE' in f][0]
    # Check if scence belongs to the heatwave
    f = h5py.File(path + lstF)
    # Extract begining datetime
    date = np.array(f['StandardMetadata']['RangeBeginningDate']).item().decode('utf-8')
    time = np.array(f['StandardMetadata']['RangeBeginningTime']).item().decode('utf-8')
    # Delete variable 
    del f
    # Combine time and date
    dateTime = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S.%f')

    # Check if dateTime belong to one of the heatwaves
    if dateInHeatwave(dateTime, heatwaves):
        # Extract the file path of the respective paths
        fileNameGeo = path + [f for f in onlyfiles if key in f and 'GEO' in f][0]
        fileNameLST = path + [f for f in onlyfiles if key in f and 'LSTE' in f][0]
        fileNameCld = path + [f for f in onlyfiles if key in f and 'CLOUD' in f][0]
        # Create the respective tifs
        #
        # createTif(fileNameGeo, fileNameLST, fileNameCld, config)
# %% Check quality of tifs in terms of shape
# Get tiff path
tiffs = [f for f in listdir(config['data']['ES_tiffs']) if f.endswith(".tif")]
shapes = []
# Loop over tiffs and print shape
for files in  tiffs:
    tif = rioxarray.open_rasterio(config['data']['ES_tiffs'] + files)
    img = np.array(tif)[0]
    if np.sum(img.shape) < 1450:
        print(img.shape)
        plt.imshow(img, cmap='jet')
        plt.show()
    shapes.append(img.shape)

# TODO: Each 'quality' output tif should have the same dimension 

# %%
# Create a Dataframe for the data quality
dataQ = pd.DataFrame(columns = ['orbitNumber', 'dateTime', 'cloudCoverage %', 'meanLSTE' ])

# Get all filepaths of the tifs
onlyfiles = [
    f 
    for f in listdir(config['data']['ES_tiffs']) 
    if isfile(join(config['data']['ES_tiffs'], f)) and f.endswith('.tif')]

# Extract all unique keys and reduce to unique values
unique_keys = set([files.split('_')[3] + '_' + files.split('_')[4] for files in onlyfiles])

# Loop over unique keys of the tif files 
for key in unique_keys:
    # Get all filepaths corresponding to the unique key
    orbitFls = [f for f in onlyfiles if key in f]
    # Open lst tiff
    lst = rioxarray.open_rasterio(
        config['data']['ES_tiffs'] + [f for f in orbitFls if 'LSTE' in f and '.tif' in f][0]
        )
    # Open cloud tiff
    cld = rioxarray.open_rasterio(
        config['data']['ES_tiffs'] + [f for f in orbitFls if 'CLOUD' in f and '.tif' in f][0]
        )
    # Fill dataQ dataframe with information about the respective tiffs
    dataQ.loc[len(dataQ)] = [
        key,
        lst.attrs['recordingTime'],
        cld.attrs['meanValue'] * 100,
        lst.attrs['meanValue']]

# Sort dataQ dataframe by time
dataQ.sort_values(by = ['dateTime'],inplace = True,ignore_index = True)

# Create a new column qualityFlag
dataQ['qualityFlag'] = (dataQ['meanLSTE'] > 0.5) & (dataQ['cloudCoverage %'] < 80)

# %% Plot LST tiff by key
key = '23127_006'
lst = rioxarray.open_rasterio(path + [f for f in [p for p in onlyfiles if key in p] if 'LSTE' in f and '.tif' in f][0])
img = np.array(lst)[0]
plt.imshow(img, cmap='jet')
plt.show()

# %% 
# Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timesplot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 15 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 17) & 
    dataQ['qualityFlag']
    ]

# %% Create masked arrays for all scenes and calculate the average
# Create an empty list
maskedArraysL = []
# Set path for geoTiffs
path = config['data']['ES_tiffs']

# Loop over tifs in afternoon
for orbitN in afterNoon['orbitNumber']:
    # Select all files from one orbit in python
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and orbitN in f]
    # Extract path of lst and cloud
    lst = rioxarray.open_rasterio(
        os.path.join(path, [f for f in files if "LSTE" in f and f.endswith(".tif")][0])
        )
    cld = rioxarray.open_rasterio(
        os.path.join(path, [f for f in files if "Cloud" in f and f.endswith(".tif")][0])
        )
    # Print shape of files
    print(lst.shape)
    print(type(lst))
    # Transform to array
    img_lst = np.array(lst)[0]
    img_cld = np.array(cld)[0]
    # Create a masked array. In addition to the cloud mask, temperature values below 1 are masked too
    masked_array = np.ma.masked_array(img_lst, mask=(img_cld.astype(bool) | (lst.values<1)))
    # Store masked arrays in a list
    maskedArraysL.append(masked_array)

# %% Create a subplot with all tiffs

# Initiate subplots
fig, axs = plt.subplots(2, 2)

# Loop over maskedArraysL
for i, ax in enumerate(axs.flat):
    ax.imshow(maskedArraysL[i], cmap='jet')
    ax.axis('off')

# Plot overall plot
plt.tight_layout()  
plt.show()

# %% Create and plot a mean array of all relevant tifs
# TODO: The Last Array doesnt fit with the shape 
# Create "mean" tif
mean_array = np.ma.mean(maskedArraysL[:-1], axis=0)

#  Plot mean array
plt.imshow(mean_array, cmap='jet')
plt.colorbar(label='Temperature')
plt.show()


# %%
# TODO: Plot tif over a interactive open street map
import contextily as ctx
from shapely.geometry import box
import folium
import matplotlib.colors as colors
# Get tif path with orbit number: 23129_012
files = [
    f 
    for f in os.listdir(config['data']['ES_tiffs']) 
    if '23129_012' in f and 'LSTE' in f and f.endswith('.tif')
    ]

# %% Import tif 
#lst = rasterio.open(config['data']['ES_tiffs'] + files[0])
lst = rioxarray.open_rasterio(config['data']['ES_tiffs'] + files[0])

# Extract values
data = np.array(lst)[0]

# Define the colormap from blue to red
cmap = plt.colormaps['jet']

# Normalize the data between 0 and 1
norm = colors.Normalize(vmin=data.min(), vmax=data.max())

# Apply the colormap to the normalized data
colored_data = cmap(norm(data))

# Set image bounds
image_bounds = box(*lst.rio.bounds())

m = folium.Map(
    location=[image_bounds.centroid.y, image_bounds.centroid.x],
    zoom_start=10,
    # control_scale=True
    )

folium.GeoJson(image_bounds.__geo_interface__).add_to(m)

# Extract bounds
min_x, min_y, max_x, max_y = lst.rio.bounds()

corner_coordinates = [[min_y, min_x], [max_y, max_x]]

# Add the OpenStreetMap tile layer with transparent colors
folium.TileLayer(
    tiles='CartoDB positron',
    attr='CartoDB',
    transparent=True,
).add_to(m)


folium.raster_layers.ImageOverlay(
        colored_data,
        bounds=corner_coordinates,
        opacity=0.4,
        interactive=True,
        cross_origin=False,
        pixelated=False,
        zindex=0.2
    ).add_to(m)

m

# TODO: Reduce map to munich 

# TODO: Add a legend

# TODO: Add water to the map


# %%
'''
# %% 
# Choose the file to plot here
# lst_path = lst_files[0]
# Extract unique identifier for the case that e.g the 6. lst-h5 does not 
# correspond to the 6. geo or cld file
# unique_idf = lst_path.replace("ECOSTRESS_L2_LSTE_", "").rsplit('_', 1)[0]
# Select the geo path and cld path that containing the uniue idf
# geo_path = [f_geo for f_geo in geo_files if unique_idf in f_geo][0]
# cld_path = [f_cld for f_cld in cloud_files if unique_idf in f_cld][0]
# Create tifs
#ws_path = "/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/raw_h5"
path = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/raw_h5'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
# %%
unique_keys = []
for files in onlyfiles:
    unique_keys.append(files.split('_')[3] + '_' + files.split('_')[4])
# Reduce on only unique values
unique_keys = set(unique_keys)
# %%
for keys in unique_keys:
    for files in onlyfiles:
# %%
#orbit_files = [f for f in onlyfiles if '23467' in f]
# %%
#tif_paths = createTif(geo_files[0], lst_files[0], cloud_files[0], config)
tif_paths = createTif(
    path + '/ECOSTRESS_L1B_GEO_23529_005_20220830T064558_0601_01.h5',
    path + '/ECOSTRESS_L2_LSTE_23529_005_20220830T064558_0601_02.h5',
    path + '/ECOSTRESS_L2_CLOUD_23529_005_20220830T064558_0601_02.h5',
    config)
# %% Plot LST tiff for munich
img_lst_MU = rasterio.open(tif_paths[0])
show(img_lst_MU)
#image_lst = Image.open(tif_paths[0].replace('.tif','_Large.png'))
#image_lst.show()
# %% Overlay over open street map
from shapely.geometry import box
dst = rioxarray.open_rasterio(tif_paths[0])
#dst = dst.rio.reproject('EPSG:4326')
image_bounds = box(*dst.rio.bounds())

# %%
img = np.array(dst)[0]
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
# %%
def colorize(array, cmap='viridis'):
    normed_data = (array - array.min()) / (array.max() - array.min())    
    cm = plt.cm.get_cmap(cmap)    
    return cm(normed_data)  

# %%
colored_data = colorize(img , cmap='viridis_r')

# %%
import folium
m = folium.Map(location=[image_bounds.centroid.y, image_bounds.centroid.x], zoom_start=14)
folium.GeoJson(image_bounds.__geo_interface__).add_to(m)
min_x, min_y, max_x, max_y = dst.rio.bounds()

corner_coordinates = [[min_y, min_x], [max_y, max_x]]
folium.raster_layers.ImageOverlay(
        colored_data,
        bounds=corner_coordinates,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=0.2,
        colormap='jet' ,
    ).add_to(m)

m




# %% Plot cloud coverage tiff
img_cld_MU = rasterio.open(tif_paths[1])
show(img_cld_MU)
#image_cld = Image.open(tif_paths[1].replace('.tif','_Large.png'))
#image_cld.show()
print(f'{np.mean(img_cld_MU.read()[0])*100:.2f}% of the picture are covered with clouds')



# %%

h5 = h5py.File('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/raw_h5/ECOSTRESS_L2_LSTE_23529_005_20220830T064558_0601_02.h5')
# %%
# %%

# %% Calculate cloud coverage
#cldX = rioxarray.open_rasterio(tif_paths[1], masked = True)
#print(f'{np.mean(img_cld.read()[0])*100:.2f}% of the picture are covered with clouds')
# %%
lst = h5py.File(lst_files[0])
# %%
lstArr = np.array(lst['SDS']['LST'])
# %% 
#img_lst = rasterio.open(tif_paths[0])
#show(img_lst)
# %%
plt.imshow(lstArr, cmap='jet')
plt.colorbar(label='Celsius')
plt.title('ECOSTRESS LST Data')
plt.show()

####################################################################################################################################
####################################################################################################################################
# %% Extract GEO file
# https://lpdaac.usgs.gov/products/eco1bgeov001/
# Store path
#geo = 'ECOSTRESS_L1B_GEO_23529_005_20220830T064558_0601_01.h5'
#lst = 'ECOSTRESS_L2_LSTE_23529_005_20220830T064558_0601_02.h5'
#lst = 'ECOSTRESS_L2_CLOUD_23529_005_20220830T064558_0601_02.h5'
i = 0
lst = ws_path+'/'+lst_files[i]
geo = ws_path+'/'+geo_files[i]
cld = ws_path+'/'+cloud_files[i]
# Read in .h5 file 
f_lst = h5py.File(lst)
# Store paths of elements in list
eco_objs = []
f_lst.visit(eco_objs.append)
# Show datasets in f_lst
lst_SDS = [str(obj) for obj in eco_objs if isinstance(f_lst[obj], h5py.Dataset)] 
#  Store path of relevant LST data
sds = ['LST','LST_err']
lst_SDS = [dataset for dataset in lst_SDS if dataset.endswith(tuple(sds))]
# 
lst_SD = f_lst[lst_SDS[0]][()]
# Set tempertature range
tempMin, tempMax = 0, 50
# Transfer it to Kelvin and scale it 
tempMin = (tempMin + 273.15) / 0.02
tempMax = (tempMax + 273.15) / 0.02
# Set "wrong values" to 0
lst_SD[(lst_SD < tempMin) | (lst_SD > tempMax)] = 0
# Create function to scale data and transfer to celcius
def kelToCel(x): # TODO: Define externally
    if np.isnan(x):
        return np.nan
    if x == 0: # MAYBE FALSE 
        return 0
    else:
        return round(((x * 0.02) - (273.15)))
# Vectorize function
kelToCel = np.vectorize(kelToCel)
# Calculate temp to celcius
lst_SD = kelToCel(lst_SD)
# %% Read in cld file
f_cld = h5py.File(cld)
# Store relative paths of elements in list
eco_objs = []
f_cld.visit(eco_objs.append)
#  Show datasets in f_lst
cld_SDS = [str(obj) for obj in eco_objs if isinstance(f_cld[obj], h5py.Dataset)]
# Store name of relevant dataset
sds = ['CloudMask']
# Extract relevant datasets
cld_SDS = [dataset for dataset in cld_SDS if dataset.endswith(tuple(sds))]
# Extrcact dataset
cld_SD = f_cld[cld_SDS[0]][()]
#  Encode the cloud coverage as function; TODO: Define externally
def get_bit(x):
    return int('{0:08b}'.format(x)[-3])
# Vectorize function
get_zero_vec = np.vectorize(get_bit)
# Apply funtion
cld_SD = get_zero_vec(cld_SD)
# %% Read in geo file
f_geo = h5py.File(geo)
# 
geo_objs = []
f_geo.visit(geo_objs.append)
# Search for lat/lon SDS inside data file
latSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/latitude' in obj]
lonSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/longitude' in obj]
# Store lat and long as numpy array
lat = f_geo[latSD[0]][()].astype(float)
lon = f_geo[lonSD[0]][()].astype(float)

# Set swath definition from lat/lon arrays
# https://pyresample.readthedocs.io/en/latest/api/pyresample.html#pyresample.geometry.SwathDefinition
# TODO: Check if its possible to cut the area here to munich
# Swath is defined by the longitude and latitude coordinates for the pixels it represents. 
# The coordinates represent the center point of each pixel
# Swathdef is defining the swath dimensions
swathDef = geom.SwathDefinition(lons=lon, lats=lat)
# Define the lat/ and long for the middle of the swath
mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]
# Define AEQD projection centered at swath center
# .Proj Performs cartographic transformations. It converts from longitude, 
# latitude to native map projection x,y coordinates and vice versa
# This projection is necessary to calculate the number of pixels?!
epsgConvert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(midLat, midLon))
# Use info from AEQD projection bbox to calculate output cols/rows/pixel size
# and convert the lower left and upper right corners of the lat/lon arrays to a 
# location (in meters) in the new projection
llLon, llLat = epsgConvert(np.min(lon), np.min(lat), inverse=False)
urLon, urLat = epsgConvert(np.max(lon), np.max(lat), inverse=False)


# Calculate the height and pixel width based on the pixel size of 70 meters
# Using the AEQD projection that displays the distance of each point to the
# center perfectly correct
areaExtent = (llLon, llLat, urLon, urLat)
cols = int(round((areaExtent[2] - areaExtent[0]) / 70))  # 70 m pixel size
rows = int(round((areaExtent[3] - areaExtent[1]) / 70))
# Define bounding box of swath based on the lat long data
llLon, llLat, urLon, urLat = np.min(lon), np.min(lat), np.max(lon), np.max(lat)
areaExtent = (llLon, llLat, urLon, urLat)
# Create area definition with estimated number of columns and rows
# Define final projection
# https://de.wikipedia.org/wiki/European_Petroleum_Survey_Group_Geodesy#EPSG-Codes
projDict = pyproj.CRS("epsg:4326")


# Define Area based on cols, rows (respective to the 70m pixel) retrieved from the AEQD projedction ; TODO: Understand
# https://pyresample.readthedocs.io/en/latest/api/pyresample.html#pyresample.geometry.AreaDefinition 
# https://pyresample.readthedocs.io/en/stable/geo_def.html#areadefinition
# geom.AreaDefinition(area_id, description, proj_id, projection, width (nr of pixel), height, area_extent)                             
areaDef = geom.AreaDefinition('4326', 'Geographic','longlat', projDict, cols, rows, areaExtent)


#  Take the smaller of the two pixel dims to determine output size and ensure square pixels
# and calculate output cols/rows with the areaExtent from the coordinates
# pixel_size_x equals the pixel height in projection units
ps = np.min([areaDef.pixel_size_x, areaDef.pixel_size_y])
cols = int(round((areaExtent[2] - areaExtent[0]) / ps))
rows = int(round((areaExtent[3] - areaExtent[1]) / ps))


# Set up a new Geographic area definition with the refined cols/rows based on the quadratic pixel size
# In contrast to the latest area defintion, just the cols and rows changed
areaDef = geom.AreaDefinition('4326', 'Geographic','longlat', projDict, cols, rows, areaExtent)
# "Problem": We have know 79'269'615 pixels but only 30'412'800 values

# Get arrays with information about the nearest neighbor to each grid point TODO:Understand
# Params: 
# - source_geo_def (Geometry definition of source)
# - target_geo_def (Geometry definition of target)
# - radius_of_influence ((Cut off distance in meters)
# - neighbours (The number of neigbours to consider for each grid point)
index, outdex, indexArr, distArr = kdt.get_neighbour_info(swathDef, areaDef, 210, neighbours=1)
# 
#ecoSD = f_lst['SDS']['CloudMask'][()]
# Encode the cloud coverage as function
#def get_bit(x):
#    return int('{0:08b}'.format(x)[-3])
# Vectorize function
#get_zero_vec = np.vectorize(get_bit)
# Apply funtion
#ecoSD = get_zero_vec(ecoSD).astype('float')

#  Read in ETinst and print out SDS attributes
#ecoSD = f_lst[lst_SDS[0]][()]
#  
#tempMin = 0
#tempMax = 50
# Transfer it to Kelvin and scale it 
#tempMin = (tempMin + 273.15) / 0.02
#tempMax = (tempMax + 273.15) / 0.02
# Set "wrong values" to 0
#ecoSD[(ecoSD < tempMin) | (ecoSD > tempMax)] = 0
# 
#def kelToCel(x):
#     if np.isnan(x):
#          return np.nan
#     else:
#         return round(((x * 0.02) - 273.15))
# Vectorize function
#kelToCel = np.vectorize(kelToCel)
# 
#ecoSD = kelToCel(ecoSD)
# 


#ecoSD = ma.masked_array(ecoSD,maskMunich)
#  Read out the attributes _FillValue, add_offset, coordys, format
#for attr in f_lst[s].attrs:
#    if type(f_lst[s].attrs[attr]) == np.ndarray:
#        print(f'{attr} = {f_lst[s].attrs[attr][0]}')
#    else:
#        print(f'{attr} = {f_lst[s].attrs[attr].decode("utf-8")}')
#  Read SDS attributes and define fill value, add offset, and scale factor if available
#try: # Works
#    fv = int(f_lst[s].attrs['_FillValue'])
#except KeyError:
#    fv = None
#except ValueError:
#    fv = f_lst[s].attrs['_FillValue'][0]
#try: # Doesnt work
#    sf = f_lst[s].attrs['_Scale'][0]
#except:
#    sf = 1
#try: # Doesnt work
#    add_off = f_lst[s].attrs['_Offset'][0]
#except:
#    add_off = 0
#try: # Works
#    units = f_lst[s].attrs['units'].decode("utf-8")
#except:
#    units = 'none'



# Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
# NOTE: This code returns a masked arrays that contain a mask to tag invalid datapoints
#LSTgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, ecoSD, index, outdex, indexArr, fill_value=None)
LSTgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, lst_SD, index, outdex, indexArr, fill_value=0)
Cldgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, cld_SD, index, outdex, indexArr, fill_value=0) # TRY fv = zero

# Define the geotransform; TODO: What is a geotransform object
gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]
#  TODO: Scaling just dont work
#LSTgeo = LSTgeo * sf + add_off            # Apply Scale Factor and Add Offset
#LSTgeo[LSTgeo == fv *  sf + add_off] = fv  # Set Fill Value;  NOTE: Fills zero with zero?! 
# Set up dictionary of arrays to export
outFiles = {'LST': LSTgeo, 'Cloud': Cldgeo}
outDir = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/geoTiff/'
outNames = []
# 
fv = np.nan
# 
#ecoName = lst.split('.h5')[0]
# %%  Loop through each item in dictionary created above
for file in outFiles:
    if file == 'LST':
        ecoName = lst.split('.h5')[0].rsplit('/')[-1]
    if file == 'Cloud':
        ecoName = cld.split('.h5')[0].rsplit('/')[-1]
    # Set up output name using output directory and filename
    outName = join(outDir, '{}_{}.tif'.format(ecoName, file))
    outNames.append(outName)
    # print("output file:\n{}\n".format(outName))
    # Get driver, specify dimensions, define and set output geotransform
    height, width = outFiles[file].shape
    # Fetchs a driver by name, here GTiff
    driv = gdal.GetDriverByName('GTiff')
    #
    dataType = gdal_array.NumericTypeCodeToGDALTypeCode(outFiles[file].dtype)
    #
    d = driv.Create(outName, width, height, 1, dataType)
    #
    d.SetGeoTransform(gt)
    # Create and set output projection, write output array data
    # Define target SRS
    srs = osr.SpatialReference()
    #
    srs.ImportFromEPSG(int('4326'))
    #
    d.SetProjection(srs.ExportToWkt())
    #
    srs.ExportToWkt()
    # Write array to band
    band = d.GetRasterBand(1)
    #
    band.WriteArray(outFiles[file])
    # Define fill value if it exists, if not, set to mask fill value
    #if fv is not None and fv != 'NaN':
    #    band.SetNoDataValue(fv)
    #else:
    #    try:
    #        band.SetNoDataValue(outFiles[file].fill_value)
    #    except:
    #        pass
    band.FlushCache()
    d, band = None, None
# %%
# Define geometries
geometries = [
    {
        'type': 'Polygon',
        'coordinates': [[
            [config.longMin, config.latMin],
            [config.longMin,config.latMax],
            [config.longMax, config.latMax],
            [config.longMax, config.latMin],
            [config.longMin, config.latMin]
            ]]}
            ]
# %% TODO: Plot a point on the geotif
def plotTiffWithCoordinats(path):
    tif_lrg = rasterio.open(path)
    # Read data
    image = tif_lrg.read(1)
    # Set Transformer
    from rasterio.transform import Affine
    transform = tif_lrg.transform
    transformer = rasterio.transform.AffineTransformer(transform)
    # Apply transformer
    row,col= transformer.rowcol(11.574110525619071, 48.138482552524245)
    # Plot LST
    if 'LST' in path:
        plt.imshow(image, cmap='jet')
    if 'Cloud' in path:
        plt.imshow(image)
    # Plot Point for Munich
    plt.scatter(col,row, color='red', marker='o')
    plt.axis('off')
    plt.savefig(outNames[0].rsplit('/')[-1].replace('.tif', '')+'_Large')
    plt.show()
# %%
plotTiffWithCoordinats(outNames[0])
# %%
plotTiffWithCoordinats(outNames[1])
# %% Loop over tif-filenames
for name in outNames:
    # Load tif
    tif = rioxarray.open_rasterio(name, masked = True)
    # Crop tif
    clipped_tif = tif.rio.clip(geometries)
    # Delete old very large tif
    os.remove(name)
    # Store new cropped tif
    clipped_tif.rio.to_raster(name)
# %%
img_lst = rasterio.open(outNames[0])
show(img_lst)
# Plot second tif (cloud coveage)
img_cld = rasterio.open(outNames[1])
show(img_cld)
# %% 
mean_temp = np.mean(img_lst.read()[0])
print(f'Mean temperature: {mean_temp:.2f}')
# %%
cloud_coverage = np.mean(img_cld.read()[0])*100
print(f'Cloud coverage: {cloud_coverage:.2f}%')
# %%
arr = img_lst.read()















# %%

# %% Open tif
fp = outName
img = rasterio.open(fp)
arr = img.read()
show(img)
# %% Plot array with colorbar
plt.imshow(arr, cmap='jet')
plt.colorbar(label='Celsius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %% 
# https://corteva.github.io/rioxarray/stable/examples/clip_geom.html
# Crop the large GeoTif to munich
lst_Tif = rioxarray.open_rasterio(outName, masked = True)
# %%
geometries = [
    {
        'type': 'Polygon',
        'coordinates': [[
            [11.7,48.0558],#[config.longMin, config.latMin],
            [11.7,48.2468],#[config.longMin,config.latMax],
            [11.7501,48.2468],#[config.longMax, config.latMax],
            [11.7501,48.0558],#[config.longMax, config.latMin],
            [11.7,48.0558]#[config.longMin, config.latMin]
        ]]
    }
]
# %% Clip to geometries
clipped = lst_Tif.rio.clip(geometries)
clipped.plot()
# %%
#clipped.rio.to_raster('munich_lst.tif')
# Plot munich
clipped.plot()
# %%
plt.imshow(img.read(), cmap='jet')
plt.colorbar(label='Celsius')
plt.title('ECOSTRESS LST Data')
plt.show()

# %%

# %% 
temp = (clipped.squeeze().values * 0.02)-273.15
# %%
plt.imshow(temp, cmap='jet')
plt.colorbar(label='Celsius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %%
# %%
temp = (lst.squeeze().values * 0.02) - 273.15
# %%
temp.rio.transform()
# %%
plt.imshow(temp, cmap='jet')
plt.colorbar(label='Celsius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %%
src = rasterio.open(outName)
plt.imshow(src.read(1), cmap='jet')
plt.show()
# %%
plt.imshow(im, cmap='jet')
plt.colorbar(label='Celcius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %% 
# %% 
geo = 'ECOSTRESS_L1B_GEO_23529_005_20220830T064558_0601_01.h5'
# Read in .h5 file 
f_geo = h5py.File(geo)
# Store longitude and latitude as matrix
lat = np.array(f_geo['Geolocation']['latitude'])
long = np.array(f_geo['Geolocation']['longitude'])
#
lst = 'ECOSTRESS_L2_LSTE_23529_005_20220830T064558_0601_02.h5'
# Read .h5 file
f_lst = h5py.File(lst)
# Store LST as array; TODO: How to transfer data to celcius? Why so many zeros?
lst = np.array(f_lst['SDS']['LST'])
lst = lst.astype(float)
# %%
# set geotransform
nx = lst.shape[0]
ny = lst.shape[1]
xmin, ymin, xmax, ymax = [long.min(), lat.min(), long.max(), lat.max()]
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)
geotransform = (xmin, xres, 0, ymax, 0, -yres)
# %%
# create the 3-band raster file
dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', ny, nx, 3, gdal.GDT_Float32)
# %%
dst_ds.SetGeoTransform(geotransform)    # specify coords
# %%
srs = osr.SpatialReference()            # establish encoding
# %%
srs.ImportFromEPSG(4326)                # WGS84 lat/long
# %%
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
# %%
dst_ds.WriteArray(lst)   # write r-band to the raster
# %%
dst_ds.FlushCache()                     # write to disk
dst_ds = None                           # save, close




# %% Store longitude and latitude as matrix
lat = np.array(f_geo['Geolocation']['latitude'])
long = np.array(f_geo['Geolocation']['longitude'])
# %% Creat Mask and combine them
latMask = np.array((lat > config.latMin)&(lat < config.latMax))
longMask = np.array((long > config.longMin)&(long < config.longMax))
# Merge both masks
maskMunich = np.array(latMask & longMask)
# %% Apply mask on lat and long
x = np.array([[1, 2, 3],[4 ,5, 6],[7, 8, 9]])
# %%
Mask = np.array([[False, False, False],[False, True, True],[False, True, True]])
# %%
np.array([[5, 6],[8, 9]])
# %% Define middel of coordinate system
mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]
# %%
mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]
# %%
epsgConvert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(midLat, midLon))





# %% Extract Land Surface temperature
# https://lpdaac.usgs.gov/products/eco2lstev001/
# Store path
lst = 'ECOSTRESS_L2_LSTE_23529_005_20220830T064558_0601_02.h5'
# Read .h5 file
f_lst = h5py.File(lst)
# Store LST as array; TODO: How to transfer data to celcius? Why so many zeros?
lst = np.array(f_lst['SDS']['LST'])
lst = lst.astype(float)
# %% Set realistic range
tempMin = -50
tempMax = 50
# Transfer it to Kelvin and scale it 
tempMin = (tempMin + 273.15) / 0.02
tempMax = (tempMax + 273.15) / 0.02
# %% Set "wrong values" to NA
lst[(lst < tempMin) | (lst > tempMax)] = np.nan
# %%
def kelToCel(x):
     if np.isnan(x):
          return np.nan
     else:
         return round(((x * 0.02) - 273.15))
# Vectorize function
kelToCel = np.vectorize(kelToCel)
# %% 
lst = kelToCel(lst)
# %% Plot the image
plt.imshow(lst, cmap='jet')
plt.colorbar(label='Celcius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %% 
lstMunich = lst*maskMunich
# %% 
import pandas as pd
df = pd.DataFrame(lstMunich)
df = df.replace(0, np.nan)
df = df.dropna(thresh=0.01*len(df.columns))
df = df.dropna(axis=1, thresh=0.01*len(df))
lstMunich = df.to_numpy()
# %% Plot the image
plt.imshow(lstMunich, cmap='jet')
plt.colorbar(label='Celcius')
plt.title('ECOSTRESS LST Data')
plt.show()
# %% Extract cloud coverage
# https://lpdaac.usgs.gov/products/eco2cldv001/
# Store file path
cld = 'ECOSTRESS_L2_CLOUD_23480_005_20220827T042037_0601_02.h5'
# Read .h5 file
f_cld = h5py.File(cld)
# %% Store cloud coverage as array
cld = np.array(f_cld['SDS']['CloudMask'])
# %% Encode the cloud coverage as function
def get_bit(x):
    return int('{0:08b}'.format(x)[-2])
# %% Vectorize function
get_zero_vec = np.vectorize(get_bit)
# %% Apply funtion
cld_mask = get_zero_vec(cld)
# %% Plot cloud coverage
plt.imshow(cld_mask,cmap = 'binary')
plt.show()
# %% This can help to get better overview over hierarchical file 
#data = {}
#
#for k in f.keys():
#    data[k] = {}
#    for sub_key in f[k].keys():
#        data[k][sub_key] = [f[k][sub_key].size, f[k][sub_key].nbytes]
# %%
'''