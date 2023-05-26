# Import
import h5py
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
import pyproj
import numpy as np
from osgeo import gdal, gdal_array, gdalconst, osr
from os.path import join
import rioxarray
import os
import requests
import json
import subprocess
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine
import datetime


# Define function to download hierarchichal files
def downloadH5(credentials, header, tempFilter, spatialFilter, config):
     ws_path = "/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/raw_h5"
     # Set api_url
     api_url = config['api']['path']
     # Set empty dictionaries for filenames
     filenames = {}
     # Store relevant datasets
     datasets = ['ecostress_eco1bgeo', 'ecostress_eco2cld', 'ecostress_eco2lste']
     #
     for dataset in datasets:
         # Define payload 
         payload = {'datasetName' : dataset,
                    'spatialFilter' : spatialFilter,
                    'acquisitionFilter':  tempFilter}
         # Requests scenes for the filtered time and area
         scenes_lste = requests.post(api_url + 'scene-search',
                                     data=json.dumps({'datasetName' : dataset,
                                                      'metadataType': 'full',
                                                      'sortDirection': 'DESC',
                                                      'sceneFilter' : payload}),
                                                      headers=header)
         # Store scenes as json
         scenes_lste = scenes_lste.json()['data']['results']
         # Create empty list for filenames
         #filenames = []
         list_name = dataset[-3:]+'_paths'
         #
         # filenames[list_name] = []
         #locals()[list_name] = []
         #
         # i = 0
         # Loop over scenes
         for scenes in scenes_lste:
             # if i == NumberofScenes:
             #     break
             # Get download options
             options = requests.post(api_url + 'download-options',
                                     data=json.dumps({'datasetName' : dataset,
                                                      'entityIds': scenes['entityId']}),
                                                      headers=header)
             # Extract HDF5
             infoH5=[dic for dic in options.json()['data'] if dic['productName'] == 'HDF5'][0]
             # Skip unaivalable scenes
             if not infoH5['available']:
                 # i += 1
                 continue
             # Request Download
             downloadRequest = requests.post(api_url + 'download-request',
                                             data=json.dumps({"downloads" : [{'entityId' : infoH5['entityId'],
                                                                              'productId': infoH5['id']}]}),
                                                                              headers=header)
             # Store URL
             url = downloadRequest.json()["data"]["availableDownloads"][0]["url"]
             # Extract filename 
             filename = url.rsplit('/',1)[1]
             # filenames[list_name].append(ws_path + '/' + filename)
             # If file already exist, dont download it again
             if os.path.exists(ws_path + '/' + filename):
                 # i += 1
                 continue
             # Set command for terminal
             command = ("wget" + " -P " + ws_path  + " --no-verbose" +  " --user=" + credentials["username"] + " --password='" + credentials["password_URS"] + "' " + url)
             # Download the data
             subprocess.run(command, shell=True,text=False)
             # Increase i for early stopping
             # i += 1
     #return filenames


def kelToCel(x):
    '''
    Converts temperature in Kelvin to Celsius.

    Parameters:
    x (float): Temperature in Kelvin.

    Returns:
    float: Temperature in Celsius.

    Notes:
    - If the input is NaN (Not a Number), the function returns NaN.
    - If the input is 0, the function returns 0.
    - For any other valid input, the function calculates the temperature in Celsius using the formula:
        Celsius = (Kelvin * 0.02) - 273.15.
        The result is rounded to the nearest whole number.
    '''
    if np.isnan(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return round(((x * 0.02) - 273.15)) # TODO: ROUNDING 


# Vectorize function
kelToCel = np.vectorize(kelToCel)



def get_bit(x):
    '''
    Encodes cloud-coverage from bit

    Parameters:
    x (int): 8-bit Numbers

    Returns:
    binary value: Cloud mask
    '''
    return int('{0:08b}'.format(x)[-3])


# Vectorize function
get_zero_vec = np.vectorize(get_bit)


# Function get path name for the geo file and the cloud lste file, produces to 
# tiff files and returns their path name; 
def createTif(fileNameGeo, fileNameLST, fileNameCld, config):
    outDir = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/geoTiff/'
    if os.path.exists(join(outDir, '{}_{}.tif'.format(fileNameLST.rsplit('/')[-1].rsplit('.h5')[0], 'LST'))):
        return
    print('Test')
    # Read in lst file
    f_lst = h5py.File(fileNameLST)
    # Extract begining datetime
    beginDate = np.array(f_lst['StandardMetadata']['RangeBeginningDate']).item().decode('utf-8')
    beginTime = np.array(f_lst['StandardMetadata']['RangeBeginningTime']).item().decode('utf-8')
    # Combine time and date
    beginDateTime = datetime.datetime.strptime(beginDate + ' ' + beginTime, '%Y-%m-%d %H:%M:%S.%f')
    # Extract ending date time
    endDate = np.array(f_lst['StandardMetadata']['RangeEndingDate']).item().decode('utf-8')
    endTime = np.array(f_lst['StandardMetadata']['RangeEndingTime']).item().decode('utf-8')
    # Combine date and time
    endDateTime = datetime.datetime.strptime(endDate + ' ' + endTime,'%Y-%m-%d %H:%M:%S.%f')
    # Calculate "mean" datetime
    recordingTime = datetime.datetime.fromtimestamp((beginDateTime.timestamp() + endDateTime.timestamp()) / 2).strftime("%Y-%m-%d %H:%M:%S")
    # Store relative paths of elements in list
    eco_objs = []
    f_lst.visit(eco_objs.append)
    #  Show datasets in f_lst
    lst_SDS = [str(obj) for obj in eco_objs if isinstance(f_lst[obj], h5py.Dataset)]
    # Store name of relevant dataset
    sds = ['LST','LST_err']
    # Extract relevant datasets
    lst_SDS  = [dataset for dataset in lst_SDS if dataset.endswith(tuple(sds))]
    # Read in data
    lst_SD = f_lst[lst_SDS[0]][()]
    # Set tempertature range
    tempMin, tempMax = 0, 50
    # Transfer it to Kelvin and scale it 
    tempMin = (tempMin + 273.15) / 0.02
    tempMax = (tempMax + 273.15) / 0.02
    # Set "wrong values" to 0
    lst_SD[(lst_SD < tempMin) | (lst_SD > tempMax)] = 0
    # NOTE: THIS PART MIGHT NOT WORK, I DIDNT TESTED IT YET
    #def kelToCel(x):
    #    if np.isnan(x):
    #        return np.nan
    #    elif x == 0:
    #        return 0
    #    else:
    #        return round(((x * 0.02) - 273.15))
    # Vectorize function
    #kelToCel = np.vectorize(kelToCel)
    # Vectorize function
    # kelToCel = np.vectorize(kelToCel)
    # Calculate temp to celcius
    lst_SD = kelToCel(lst_SD)
    # Read in lst file
    f_cld = h5py.File(fileNameCld)
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
    #def get_bit(x):
    #    return int('{0:08b}'.format(x)[-3])
    # Vectorize function
    #get_zero_vec = np.vectorize(get_bit)
    # Apply funtion
    cld_SD = get_zero_vec(cld_SD)
    # Read in .h5 file 
    f_geo = h5py.File(fileNameGeo)
    # Store relative paths in geo h5 file
    geo_objs = []
    f_geo.visit(geo_objs.append)
    # Search for lat/lon SDS inside data fil
    latSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/latitude' in obj]
    lonSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/longitude' in obj]
    # Store lat and long as numpy array
    lat = f_geo[latSD[0]][()].astype(float)
    lon = f_geo[lonSD[0]][()].astype(float)
    # Set swath definition from lat/lon arrays
    # https://pyresample.readthedocs.io/en/latest/api/pyresample.html#pyresample.geometry.SwathDefinition
    # TODO: Check if its possible to cut at this point to munich
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
    # Take the smaller of the two pixel dims to determine output size and ensure square pixels
    # and calculate output cols/rows with the areaExtent from the coordinates
    # pixel_size_x equals the pixel height in projection units
    ps = np.min([areaDef.pixel_size_x, areaDef.pixel_size_y])
    cols = int(round((areaExtent[2] - areaExtent[0]) / ps))
    rows = int(round((areaExtent[3] - areaExtent[1]) / ps))
    # Set up a new Geographic area definition with the refined cols/rows based on the quadratic pixel size
    # In contrast to the latest area defintion, just the cols and rows changed
    areaDef = geom.AreaDefinition('4326', 'Geographic','longlat', projDict, cols, rows, areaExtent)
    # "Problem": We have know 79'269'615 pixels but only 30'412'800
    # Get arrays with information about the nearest neighbor to each grid point TODO:Understand
    # Params: 
    # - source_geo_def (Geometry definition of source)
    # - target_geo_def (Geometry definition of target)
    # - radius_of_influence ((Cut off distance in meters)
    # - neighbours (The number of neigbours to consider for each grid point)
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(swathDef, areaDef, 210, neighbours=1)
    # Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
    # NOTE: This code returns a masked arrays that contain a mask to tag invalid datapoints
    LSTgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, lst_SD, index, outdex, indexArr, fill_value=0)
    Cldgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, cld_SD, index, outdex, indexArr, fill_value=0) # TRY fv = zero
    #  Define the geotransform; TODO: What is a geotransform object
    gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]
    # Set up dictionary of arrays to export
    outFiles = {'LST': LSTgeo, 'Cloud': Cldgeo}
    outDir = '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/ECOSTRESS/geoTiff/'
    outNames = []
    # Set fill value
    fv = np.nan
    # Define geometries
    geometries = [
        {
            'type': 'Polygon',
            'coordinates': [[
                [config['bboxes']['munich'][0], config['bboxes']['munich'][1]],
                [config['bboxes']['munich'][0], config['bboxes']['munich'][3]],
                [config['bboxes']['munich'][2], config['bboxes']['munich'][3]],
                [config['bboxes']['munich'][2], config['bboxes']['munich'][1]],
                [config['bboxes']['munich'][0], config['bboxes']['munich'][1]]
                ]]}
                ]
    # ecoName = lst.split('.h5')[0]
    for file in outFiles:
        if file == 'LST':
            ecoName = fileNameLST.rsplit('/')[-1].rsplit('.h5')[0]
        if file == 'Cloud':
            ecoName = fileNameCld.rsplit('/')[-1].rsplit('.h5')[0]
        # Set up output name using output directory and filename
        outName = join(outDir, '{}_{}.tif'.format(ecoName, file))
        # print(outName)
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
        #
        band.FlushCache()
        #
        d, band = None, None
    # Loop over tif-filenames
    #for name in outNames:  NOTE: THIS PART MIGHT NOT WORK, I DIDNT TESTED IT YET
        # Plot a png
        plotTiffWithCoordinats(outName)
        # Load tif
        tif = rioxarray.open_rasterio(outName, masked = True)
        # Crop tif
        clipped_tif = tif.rio.clip(geometries)
        # Mean temp but only for lst: round(np.mean(tif.data),4)
        clipped_tif.attrs['meanValue'] = round(np.mean(clipped_tif.data), 6)
        # Time as attribute
        clipped_tif.attrs['recordingTime'] = recordingTime
        # Delete old very large tif
        os.remove(outName)
        # Store new cropped tif
        clipped_tif.rio.to_raster(outName)


def plotTiffWithCoordinats(path):
    '''
    Plots a PNG image with a given raster file and a marker for Munich.

    Parameters:
    path (str): The file path of the raster file.

    Returns:
    None

    Raises:
    FileNotFoundError: If the specified file path does not exist.

    '''
    try:
        # Open the TIFF file
        tif_lrg = rasterio.open(path)
    except FileNotFoundError:
        raise FileNotFoundError("The specified file path does not exist.")

    # Read data
    image = tif_lrg.read(1)
    # Set Transformer 
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
    plt.savefig(path.replace('.tif', '') + '_Large')
     # Close the plot
    plt.close()


def dateInHeatwave(date, heatwaves):
    '''
    Checks if a given date falls within any of the heatwave periods.

    Args:
        date (datetime): The date to check.
        heatwaves (list): A list of heatwave dictionaries, each containing 
        'start' and 'end' keys specifying the start and end dates of a heatwave.

    Returns:
        bool: True if the date falls within a heatwave, False otherwise.
    '''
    for wave in heatwaves:
        start_date = datetime.datetime.strptime(wave['start'], '%Y-%m-%d %H:%M:%S')
        end_date = datetime.datetime.strptime(wave['end'], '%Y-%m-%d %H:%M:%S')
        if start_date <= date <= end_date:
            return True
    
    return False


def heatwave_transform(dates):
    '''
    Transforms a list of dates into heatwave intervals.

    Args:
        dates (list): A list of datetime objects representing dates.

    Returns:
        list: A list of dictionaries, each containing the start and end dates of a heatwave.
    '''
    heatwaves = []
    start_date = None
    end_date = None

    for date in sorted(dates):
        if start_date is None:
            start_date = date
            end_date = date
        elif date - end_date == datetime.timedelta(days=1):
            end_date = date
        else:
            heatwaves.append({'start': start_date.strftime('%Y-%m-%d 00:00:00'), 'end': end_date.strftime('%Y-%m-%d 23:59:00')})
            start_date = date
            end_date = date
    # Append the last heatwave
    heatwaves.append({'start': start_date.strftime('%Y-%m-%d 00:00:00'), 'end': end_date.strftime('%Y-%m-%d 23:59:00')})

    return heatwaves