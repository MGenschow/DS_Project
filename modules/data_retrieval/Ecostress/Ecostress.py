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
import statistics
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

# %% Import heatwaves; TODO: Understand definition of heatwaves
# TODO: Is it possible to soften the definition?
dates = pd.read_pickle('/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/DWD/heatwaves.pkl')

# Combine dates to periods format of heatwaves
heatwaves = heatwave_transform(dates)

heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-21 23:59:00'})
heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-15 23:59:00'})

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
confirmation = input("Do you want to download the hierarchical files (Y/n): ")
if confirmation.lower() == "y":
    for temporalFilter in heatwaves:
        downloadH5(credentials, headers, temporalFilter, spatialFilter, config)
else:
    print("Loop execution cancelled.")

# %% Extract unique keys and create a tiff for each unique scene 
# Extract all unique keys
# Get filepaths of all h5 files
onlyfiles = [
    f 
    for f in listdir(config['data']['ES_raw']) 
    if isfile(join(config['data']['ES_raw'], f))]
    
# Extract keys
keys = [files.split('_')[3] + '_' + files.split('_')[4] for files in onlyfiles if 'LSTE' in files]
# Reduce to unique
unique_keys = set(keys)

# Delete tiff files
confirmation = input("Do you really want to delete all tiff files (Y/n): ")
if confirmation.lower() == "y":
    files = os.listdir(config['data']['ES_tiffs'])
    for file in files:
        os.remove(os.path.join(config['data']['ES_tiffs'], file))
else:
    print("No tiffs were deleted.")

# Create tif for all files corresponding to the heatwaves
path = config['data']['ES_raw']
count = 0

# Loop over all unique keys in the raw_h5 folder
for key in unique_keys:

    # Check if all files are aivalable
    if len([f for f in onlyfiles if key in f]) != 3:
        print(f'There are files missing for the key: {key}')
        continue

    # Get file path for the lst file
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

    if dateInHeatwave(dateTime, heatwaves):
        
        # Extract the file path of the respective paths
        fileNameGeo = path + [f for f in onlyfiles if key in f and 'GEO' in f][0]
        fileNameLST = path + [f for f in onlyfiles if key in f and 'LSTE' in f][0]
        fileNameCld = path + [f for f in onlyfiles if key in f and 'CLOUD' in f][0]
        
        # Create the respective tifs
        createTif(fileNameGeo, fileNameLST, fileNameCld, config)


# %% Create a Dataframe to check the quality of all relevant tiffs
# Create empty data frame
dataQ = pd.DataFrame(
    columns = [
        'orbitNumber',
        'dateTime',
        'cloudCoverage in %',
        'meanLSTE' ])

# Get all filepaths of the relevant tiffs
onlyfiles = [
    f 
    for f in listdir(config['data']['ES_tiffs']) 
    if isfile(join(config['data']['ES_tiffs'], f)) and f.endswith('.tif')]

# Extract all unique keys and reduce to unique values
unique_keys = set(
    [files.split('_')[3] + '_' + files.split('_')[4] for files in onlyfiles]
    )

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
# Create a new column qualityFlag based on average temperature and cloud coverage
dataQ['qualityFlag'] = (dataQ['meanLSTE'] > 0.5) & (dataQ['cloudCoverage in %'] < 80)

# %% Plot LST tiff by key
key = '23129_012'
lst = rioxarray.open_rasterio(
    config['data']['ES_tiffs'] + [f for f in [p for p in onlyfiles if key in p] if 'LSTE' in f and '.tif' in f][0]
    )
img = np.array(lst)[0]
plt.imshow(img, cmap='jet')
plt.show()

# %% 
# Create DF with relevant tifs for the afternoon
afterNoon = dataQ[
    # TODO: How to choose the timeslot ?
    (pd.to_datetime(dataQ['dateTime']).dt.hour >= 15 ) & 
    (pd.to_datetime(dataQ['dateTime']).dt.hour <= 17) & 
    dataQ['qualityFlag']
    ]

#  %% TODO: Create function
# Create empty lists
maskedArraysL = []
pixel_sizes = []
bounding_boxes = [] 

# Set path for geoTiffs
path = config['data']['ES_tiffs']

# Loop over tiffs in afternoon and store them as masked array
for orbitN in afterNoon['orbitNumber']:

    # Select all files from one orbit
    files=[
        f 
        for f in os.listdir(path) 
        if os.path.isfile(os.path.join(path, f)) and orbitN in f
        ]

    # Extract path of lst and cloud
    lst=rasterio.open(
        os.path.join(path, [f for f in files if "LSTE" in f and f.endswith(".tif")][0])
        )
    cld=rasterio.open(
        os.path.join(path, [f for f in files if "Cloud" in f and f.endswith(".tif")][0])
        )
    # Store the pixel size of the picture in a list    
    pixel_sizes.append((lst.transform[0], lst.transform[4]))
    # Store the bounding boxes in a list
    bounding_boxes.append(lst.bounds)
    
    # Deal with faulty data
    if lst.shape != cld.shape:
        raise ValueError("Array shapes of lst and cld are not equal.")
    
    elif abs(lst.shape[0]-643) > 10 or abs(lst.shape[1]-866) > 10:
        raise ValueError("Array shape deviates too much from the desired size")

    elif lst.shape == (643, 866):
        
        # Transform to array
        img_lst = lst.read()[0]
        img_cld = cld.read()[0]
        
        # Create a masked array. In addition to the cloud mask, temperature values below 1 are masked too
        masked_array = np.ma.masked_array(img_lst, mask=(img_cld.astype(bool) | (lst.read()[0]<1)))
        
        # Store masked arrays in a list
        maskedArraysL.append(masked_array)
    
    else:

        # Tranform the pixel to the desired shape of 643x866
        lst_transformed = lst.read(
            out_shape=(lst.count, 643, 866), resampling=Resampling.bilinear
            )[0]
        cld_transformed = cld.read(
            out_shape=(cld.count, 643, 866), resampling=Resampling.bilinear
            )[0]

        # Store arrays as masked arrey
        masked_array = np.ma.masked_array(lst_transformed, mask=(cld_transformed.astype(bool) | (lst_transformed<1)))
        # Store masked arrays in a list
        maskedArraysL.append(masked_array)

# Calculate the average for each respective pixel
mean_array = np.ma.mean(maskedArraysL, axis=0)

# %% Store mean array as tiff

# Set pixel size for geo tiff
pixel_size = [statistics.mean(values) for values in zip(*pixel_sizes)]

# Set values for the bounding box
left = np.mean([box.left for box in bounding_boxes])
bottom = np.mean([box.bottom for box in bounding_boxes])
right = np.mean([box.right for box in bounding_boxes])
top = np.mean([box.top for box in bounding_boxes])

# Create bounding box
boundingBox = rasterio.coords.BoundingBox(left, bottom, right, top)
xmin, ymin, xmax, ymax = boundingBox

# Set geotransform
geotransform = (xmin, pixel_size[0], 0, ymax, 0, pixel_size[1])

# Create geotiff
array_to_tiff(mean_array, 'mean_afternoon.tif' , geotransform)

# %% 
tif = rasterio.open('mean_afternoon.tif')
plt.imshow(tif.read()[0],'jet')

# %% 
plt.imshow(mean_array,'jet')



# %% Create a subplot with all tiffs
# Initiate subplots TODO: Make this code flexible
fig, axs = plt.subplots(2, 2)

# Loop over maskedArraysL
for i, ax in enumerate(axs.flat):
    ax.imshow(maskedArraysL[i], cmap='jet')
    ax.axis('off')

# Plot overall plot
plt.tight_layout()  
plt.show()


# %% TODO: As function
# Plot tif over a interactive open street map
array_to_foliumMap('mean_afternoon.tif')

# TODO: Plot the mean tiff
# TODO: Reduce map to munich 
# TODO: Add a legend
# TODO: Add water to the map

# %%
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
# 
'''
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
#
'''