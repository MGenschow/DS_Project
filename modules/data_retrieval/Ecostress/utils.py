# Import packages
import h5py
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
import pyproj
import numpy as np
from osgeo import gdal, gdal_array, gdalconst, osr
from os.path import join, isfile
import rioxarray
import os
import requests
import json
import subprocess
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine
import datetime
import matplotlib.colors as colors
from shapely.geometry import box
import folium
from rasterio.enums import Resampling
from os import listdir
import pandas as pd
import statistics
import branca.colormap as cm
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from scipy.ndimage import binary_dilation



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
        elif date - end_date == timedelta(days=1):
            end_date = date
        else:
            end_date += timedelta(days=1)
            # end_date = end_date.replace(hour=0, minute=0, second=0)

            heatwaves.append({'start': start_date.strftime('%Y-%m-%d 00:00:00'), 'end': end_date.strftime('%Y-%m-%d 00:00:00')})
            start_date = date
            end_date = date
    # Append the last heatwave
    heatwaves.append({'start': start_date.strftime('%Y-%m-%d 00:00:00'), 'end': end_date.strftime('%Y-%m-%d 00:00:00')})

    # Add heatwaves from 2021
    heatwaves.append({'start': '2021-06-17 00:00:00', 'end': '2021-06-22 00:00:00'})
    heatwaves.append({'start': '2021-08-13 00:00:00', 'end': '2021-08-16 00:00:00'})

    return heatwaves


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
        start_date = datetime.strptime(wave['start'], '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(wave['end'], '%Y-%m-%d %H:%M:%S')
        if start_date <= date <= end_date:
            return True
    
    return False


def downloadH5(credentials, header, tempFilter, spatialFilter, config):
     '''
     Downloads hierarchical files for specified datasets based on given filters.
 
     Parameters:
     - credentials (dict): A dictionary containing the authentication credentials.
         - username (str): The username for authentication.
         - password_URS (str): The password for authentication.
     - header (dict): A dictionary containing the headers for HTTP requests.
     - tempFilter (dict): A dictionary specifying the temporal filter.
     - spatialFilter (dict): A dictionary specifying the spatial filter.
     - config (dict): A dictionary containing the configuration settings.
         - data (dict): A dictionary containing the paths for data storage.
             - ES_raw (str): The workspace path for storing the downloaded files.
         - api (dict): A dictionary containing the API settings.
             - path (str): The URL of the API.

     Returns:
     None
     '''

     # Set workspace path
     ws_path = config['data']['ES_raw']
     # Set api_url
     api_url = config['api']['path']

     # Store relevant datasets
     datasets = ['ecostress_eco1bgeo', 'ecostress_eco2cld', 'ecostress_eco2lste']

     # Loop over the datasets
     for dataset in datasets:

         # Define payload 
         payload = {
            'datasetName' : dataset,
            'spatialFilter' : spatialFilter,
            'acquisitionFilter':  tempFilter
            }
        
         # Requests scenes for the filtered time and area
         scenes_lste = requests.post(api_url + 'scene-search',
                                     data=json.dumps({'datasetName' : dataset,
                                                      'metadataType': 'full',
                                                      'sortDirection': 'DESC',
                                                      'sceneFilter' : payload}),
                                                      headers=header)

         # Store scenes as json
         scenes_lste = scenes_lste.json()['data']['results']

         # Loop over scenes
         for scenes in scenes_lste:

             # Get download options
             options = requests.post(api_url + 'download-options',
                                     data=json.dumps({'datasetName' : dataset,
                                                      'entityIds': scenes['entityId']}),
                                                      headers=header)

             # Extract HDF5
             infoH5 = [dic for dic in options.json()['data'] if dic['productName'] == 'HDF5'][0]

             # Skip unaivalable scenes
             if not infoH5['available']:
                 print('Scene not available.')
                 continue

             # Request Download
             downloadRequest = requests.post(api_url + 'download-request',
                                             data=json.dumps(
                                                {"downloads": [{'entityId' : infoH5['entityId'], 'productId': infoH5['id']}]}
                                                            ),
                                                headers=header)

             # Store URL
             url = downloadRequest.json()["data"]["availableDownloads"][0]["url"]
             # Extract filename 
             filename = url.rsplit('/',1)[1]

             # If file already exist, dont download it again
             if os.path.exists(ws_path + filename):
                 continue

             # Set command for terminal
             command = (
                 "wget" + 
                 " -P " + 
                 ws_path  + 
                 " --no-verbose" +
                 " --user=" + 
                 credentials["username"] +
                 " --password='" +
                 credentials["password_URS"] +
                 "' " +
                 url
                 )
             
             # Download the data
             subprocess.run(command, shell=True,text=False)



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
        return round(((x * 0.02) - 273.15), 4) # TODO: ROUNDING 


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


def createTif(fileNameGeo, fileNameLST, fileNameCld, temperatureRange, config):
    '''
    Process HDF5 files and convert them to GeoTIFF files.

    Args:
        fileNameGeo (str): Path to the geographical HDF5 file.
        fileNameLST (str): Path to the LST (Land Surface Temperature) HDF5 file.
        fileNameCld (str): Path to the cloud HDF5 file.
        config (dict): Configuration settings.

    Returns:
        None
    '''
    # Specify directory for the tiffs
    outDir = config['data']['ES_tiffs']

    try:
        # Read in lst file
        f_lst = h5py.File(fileNameLST)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return


    # Extract begining datetime
    beginDate = np.array(f_lst['StandardMetadata']['RangeBeginningDate']).item().decode('utf-8')
    beginTime = np.array(f_lst['StandardMetadata']['RangeBeginningTime']).item().decode('utf-8')
    # Combine time and date
    beginDateTime = datetime.strptime(beginDate + ' ' + beginTime, '%Y-%m-%d %H:%M:%S.%f')

    # Extract ending date time
    endDate = np.array(f_lst['StandardMetadata']['RangeEndingDate']).item().decode('utf-8')
    endTime = np.array(f_lst['StandardMetadata']['RangeEndingTime']).item().decode('utf-8')
    # Combine date and time
    endDateTime = datetime.strptime(endDate + ' ' + endTime,'%Y-%m-%d %H:%M:%S.%f')

    # Calculate "mean" datetime
    recordingTime = (
        datetime.
        fromtimestamp((beginDateTime.timestamp() + endDateTime.timestamp()) / 2).
        strftime("%Y-%m-%d %H:%M:%S")
        )
    
    # Store relative paths of elements in list
    eco_objs = []
    f_lst.visit(eco_objs.append)

    # Show datasets in f_lst
    lst_SDS = [str(obj) for obj in eco_objs if isinstance(f_lst[obj], h5py.Dataset)]

    # Store name of relevant dataset
    sds = ['LST']
    # Extract relevant datasets
    lst_SDS  = [dataset for dataset in lst_SDS if dataset.endswith(tuple(sds))]

    # Read in data
    lst_SD = f_lst[lst_SDS[0]][()].astype(np.float32)

    # Set max and min values
    tempMin, tempMax = temperatureRange[beginDateTime.month]

    # Set tempertature range
    # tempMin, tempMax = 0, 50
    # Transfer it to Kelvin and scale it 
    tempMin = (tempMin + 273.15) / 0.02
    tempMax = (tempMax + 273.15) / 0.02

    # Set "wrong values" to 0; NOTE: This might lead to strange errors
    lst_SD[(lst_SD < tempMin) | (lst_SD > tempMax)] =np.nan

    # Calculate temp to celcius
    lst_SD = kelToCel(lst_SD)

    try:
        # Read in lst file
        f_cld = h5py.File(fileNameCld)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return

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

    # Apply funtion
    cld_SD = get_zero_vec(cld_SD).astype(np.float32)


    # Read in .h5 file 
    try:
        f_geo = h5py.File(fileNameGeo)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    # Store relative paths in geo h5 file
    geo_objs = []
    f_geo.visit(geo_objs.append)
    # Search for lat/lon SDS inside data fil
    latSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/latitude' in obj]
    lonSD = [str(obj) for obj in geo_objs if isinstance(f_geo[obj], h5py.Dataset) and '/longitude' in obj]

    # Store lat and long as numpy array
    lat = f_geo[latSD[0]][()].astype(float)
    lon = f_geo[lonSD[0]][()].astype(float)

    # Set swath definition/ dimension from lat/lon arrays
    swathDef = geom.SwathDefinition(lons=lon, lats=lat)

    # Define the lat/ and long for the middle of the swath
    mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
    midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]

    # Define AEQD projection centered at swath center
    # .Proj Performs cartographic transformations. It converts from longitude, 
    # latitude to native map projection x,y coordinates and vice versa
    # This projection is necessary to calculate the number of pixels
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

    # Define final projection
    projDict = pyproj.CRS("epsg:4326")

    # Define Area based on cols, rows (respective to the 70m pixel) retrieved from the AEQD projedction                         
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

    # Get arrays with information about the nearest neighbor to each grid point
    # Params: 
    # - source_geo_def (Geometry definition of source)
    # - target_geo_def (Geometry definition of target)
    # - radius_of_influence ((Cut off distance in meters)
    # - neighbours (The number of neigbours to consider for each grid point)
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(swathDef, areaDef, 210, neighbours=1)

    # Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
    # NOTE: This code returns a masked arrays that contain a mask to tag invalid datapoints
    LSTgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, lst_SD, index, outdex, indexArr, fill_value=np.nan)
    Cldgeo = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, cld_SD, index, outdex, indexArr, fill_value=np.nan)

    #  Define the geotransform
    gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]

    # Set up dictionary of arrays to export
    outFiles = {'LST': LSTgeo, 'Cloud': Cldgeo}
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
                ]]
                }
                ]
    
    # Loop over outfiles
    for file in outFiles:
        
        # Define output name
        if file == 'LST':
            ecoName = fileNameLST.rsplit('/')[-1].rsplit('.h5')[0]
        elif file == 'Cloud':
            ecoName = fileNameCld.rsplit('/')[-1].rsplit('.h5')[0]

        # Set up output name using output directory and filename
        outName = join(outDir, '{}_{}.tif'.format(ecoName, file))

        #if os.path.exists(outName):
        #    continue

        # Call array to tiff function
        array_to_tiff(outFiles[file], outName, gt)

        # Plot a png
        plotTiffWithCoordinats(outName)

        # Load tif
        tif = rioxarray.open_rasterio(outName, masked = True)
        
        # Crop tif
        clipped_tif = tif.rio.clip(geometries) # all_touched = True)
        
        # Mean temp but only for lst: round(np.mean(tif.data),4)
        clipped_tif.attrs['meanValue'] = round(np.nanmean(clipped_tif.data), 6)
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


def array_to_tiff(file, outputDir, geoTrans):
    '''
    Convert a NumPy array to a GeoTIFF file.

    Args:
        file (numpy.ndarray): Input array to be converted.
        outputDir (str): Output directory and filename for the GeoTIFF file.
        geoTrans (tuple): Geotransform parameters specifying the spatial referencing.

    Returns:
        None
    '''
     # Get driver, specify dimensions, define and set output geotransform
    height, width = file.shape
    # Initiate driver
    driv = gdal.GetDriverByName('GTiff')
    # Set datatype
    dataType = gdal_array.NumericTypeCodeToGDALTypeCode(file.dtype)
    # Specify driver
    d = driv.Create(outputDir, width, height, 1, dataType)
    # Set geo transform
    d.SetGeoTransform(geoTrans)
    # Create and set output projection, write output array data
    # Define target SRS
    srs = osr.SpatialReference()
    # Set projection
    srs.ImportFromEPSG(int('4326'))
    # Export projection
    d.SetProjection(srs.ExportToWkt())
    srs.ExportToWkt()
    # Array to band
    band = d.GetRasterBand(1)
    # Write tif to array
    band.WriteArray(file)

    band.FlushCache()
    d, band = None, None


def processHF(timePeriod, temperatureRange, config):
    '''
    Processes heatwave data by creating TIFF files for 
    corresponding HDF5 files and deleting existing TIFF files if confirmed.

    Parameters:
    - timePeriod (list): List of heatwave data.
    - config (dict): A dictionary containing configuration information.

    Returns:
    - None
    '''
    # Get filepaths of all h5 files
    onlyfiles = [
        f 
        for f in listdir(config['data']['ES_raw']) 
        if isfile(join(config['data']['ES_raw'], f))
     ]
    
    # Extract keys
    keys = [
        files.split('_')[3] + '_' + files.split('_')[4] 
        for files in onlyfiles if 'LSTE' in files
     ]
    # Reduce to unique keys
    unique_keys = set(keys)
    print(f'There are {len(unique_keys)} unique keys')
    
    # Create tif for all files corresponding to the timePeriod
    path = config['data']['ES_raw']

    # Loop over all unique keys in the raw_h5 folder
    for key in unique_keys:
        
        # Check if all files are aivalable
        if len([f for f in onlyfiles if key in f]) != 3:
            print(f'There are files missing for the key: {key}')
            continue

        # Get file path for the lst file
        lstF = [f for f in onlyfiles if key in f and 'LSTE' in f][0]

        if os.path.exists(config['data']['ES_tiffs'] + lstF.replace('.h5','_LST.tif')):
            print('File does already exist.')
            continue
        
        else:
            # Check if scence belongs to the heatwave
            f = h5py.File(path + lstF)
            # Extract begining datetime
            date = np.array(f['StandardMetadata']['RangeBeginningDate']).item().decode('utf-8')
            time = np.array(f['StandardMetadata']['RangeBeginningTime']).item().decode('utf-8')
            # Delete variable 
            del f
            # Combine time and date
            dateTime = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S.%f')

            if dateInHeatwave(dateTime, timePeriod):
        
                # Extract the file path of the respective paths
                fileNameGeo = path + [f for f in onlyfiles if key in f and 'GEO' in f][0]
                fileNameLST = path + [f for f in onlyfiles if key in f and 'LSTE' in f][0]
                fileNameCld = path + [f for f in onlyfiles if key in f and 'CLOUD' in f][0]
                print('Start creating a new tif')
                # Create the respective tifs
                createTif(fileNameGeo, fileNameLST, fileNameCld, temperatureRange, config)

            else:
                print('Scence does not fall into the heat period')
                continue


def dataQualityOverview(heatPeriods, tempThresh, cloudCoverage, config):
    '''
    Generates an overview of data quality from TIFF files based on 
    the provided configuration.

    Parameters:
    - config (dict): A dictionary containing configuration information.

    Returns:
    - dataQ (pandas.DataFrame): Data frame containing information about 
      data quality, including orbit number, date and time, 
      cloud coverage percentage, and mean land surface temperature.
    '''

    final_shape = (643, 866)

    # Create empty data frame
    dataQ = pd.DataFrame(
        columns = [
            'orbitNumber',
            'dateTime',
            'cloudCoverage in %',
            'meanLSTE',
            'ratio',
            'closeToFS',
            'uniqueLSTValues'
            ])

    # Get all filepaths of the relevant tiffs, TODO: Check if tif in heatwave
    onlyfiles = [
        f 
        for f in listdir(config['data']['ES_tiffs']) 
        if isfile(join(config['data']['ES_tiffs'], f)) and 
        f.endswith('.tif') and dateInHeatwave(datetime.strptime(f.split('_')[5], '%Y%m%dT%H%M%S'), heatPeriods)
        ]

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
            config['data']['ES_tiffs'] + [f for f in orbitFls if 'LSTE' in f and f.endswith('.tif')][0]
            )
        ratio = max(lst.shape[1:]) / min(lst.shape[1:])

        differFinalShape = (abs(lst.shape[1]-final_shape[0]) < 10 or abs(lst.shape[2]-final_shape[1]) < 10)

        uniqueValues = len(np.unique(lst.values.flatten()))

        # Open cloud tiff
        cld = rioxarray.open_rasterio(
            config['data']['ES_tiffs'] + [f for f in orbitFls if 'CLOUD' in f and '.tif' in f][0]
            )

        try:
            mean_value = cld.attrs['meanValue'] * 100
        except KeyError:
            mean_value = np.nan
        # Fill dataQ dataframe with information about the respective tiffs
        dataQ.loc[len(dataQ)] = [
            key,
            lst.attrs['recordingTime'],
            mean_value,
            lst.attrs['meanValue'],
            ratio,
            differFinalShape,
            uniqueValues]

    # Sort dataQ dataframe by time
    dataQ.sort_values(by=['dateTime'], inplace=True, ignore_index=True)
    # Create a new column qualityFlag based on average temperature and cloud coverage
    dataQ['qualityFlag'] = (dataQ['meanLSTE'] > tempThresh) & \
        (dataQ['cloudCoverage in %'] < cloudCoverage) & \
        (dataQ['ratio'] < 1.5) & \
        (abs(0-dataQ['meanLSTE']) > 0.001) & \
        dataQ['closeToFS'] & \
        (dataQ['uniqueLSTValues'] > 250) & \
        (dataQ['orbitNumber'] != '23388_006')

    return dataQ


def meanMaskArray(orbitNumbers, config):
    '''
    Calculate the average for each respective pixel from a set of GeoTIFF files,
    applying masks based on cloud coverage and temperature values.

    Args:
        orbitNumbers (list): List of orbit numbers to process.
        config (dict): Configuration dictionary containing data paths.

    Returns:
        tuple: A tuple containing the following elements:
            - mean_array (numpy.ma.core.MaskedArray): A masked array representing 
              the mean values for each pixel.
            - maskedArraysL (list): A list of masked arrays for each input GeoTIFF file.
            - pixel_sizes (list): A list of tuples representing the pixel sizes for 
              each GeoTIFF file.
            - bounding_boxes (list): A list of bounding boxes for each GeoTIFF file.
    '''
    final_shape = (643, 866)
    # Create empty lists
    maskedArraysL = []
    pixel_sizes = []
    bounding_boxes = []

    # Set path for geoTiffs
    path = config['data']['ES_tiffs']

    # Loop over tiffs in afternoon and store them as masked array
    for orbitN in orbitNumbers:

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
    
        # Deal with faulty data final_shape[0]
        if lst.shape != cld.shape:
            raise ValueError("Array shapes of lst and cld are not equal.")
    
        elif abs(lst.shape[0]-final_shape[0]) > 10 or abs(lst.shape[1]-final_shape[1]) > 10:
            print("Array shape deviates too much from the desired size")
            continue
            #raise ValueError("Array shape deviates too much from the desired size")


        elif lst.shape == final_shape:
            # Transform to array
            img_lst = lst.read()[0]
            img_cld = cld.read()[0]

            # Perform dilation
            img_cld = binary_dilation(img_cld, iterations=2)
            
            # TODO: Add quality control
            # Create a masked array. In addition to the cloud mask, temperature values below 1 are masked too
            masked_array = np.ma.masked_array(img_lst, mask=(img_cld.astype(bool) | np.isnan(img_lst)))
            # Store masked arrays in a list
            maskedArraysL.append(masked_array)
    
        else:
            # Tranform the pixel to the desired shape of 643x866
            lst_transformed = lst.read(
                out_shape=(lst.count, final_shape[0], final_shape[1]), resampling=Resampling.bilinear
                )[0]
            cld_transformed = cld.read(
                out_shape=(cld.count, final_shape[0], final_shape[1]), resampling=Resampling.bilinear
                )[0]
            
            # Perform dilation
            cld_transformed = binary_dilation(cld_transformed, iterations=2)

            # Store arrays as masked arrey
            masked_array = np.ma.masked_array(lst_transformed, mask=(cld_transformed.astype(bool) | np.isnan(lst_transformed)))
            # Store masked arrays in a list
            maskedArraysL.append(masked_array)

    # Calculate the average for each respective pixel
    mean_array = np.ma.mean(maskedArraysL, axis=0)
    
    return mean_array, maskedArraysL, pixel_sizes, bounding_boxes


def mergeTiffs(orbitNrs, path, config):
    '''
    Merges multiple tiffs into a single GeoTIFF file.

    Parameters:
        orbitNrs (list): List of orbit numbers.
        path (str): Path for storing the output GeoTIFF.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing the mean array and a list of masked arrays.
    '''
    # Calculate mean masked array
    output = meanMaskArray(orbitNrs, config)
    # Store output in variables
    mean_array, maskedArraysL, pixel_sizes, bounding_boxes = output

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

    if '/' in path: 
        outDir = path
    else:
        # Create geotiff
        outDir = config['data']['ES_tiffs'].replace('geoTiff/', '') + path
    
    # Store mean array as tiff
    array_to_tiff(mean_array, outDir, geotransform)

    return mean_array, maskedArraysL


def arrays_subplot(masked_array_list):
    '''
    Plots a grid of masked arrays using subplots.

    Parameters:
        masked_array_list (list of numpy.ndarray): List of 2D masked arrays to be plotted.

    Returns:
        None
    '''
    # Get number of arrays
    num_plots = len(masked_array_list)
    # Calculate number of rows and cols
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))

    # Initiate subplot
    fig, axs = plt.subplots(rows, cols)

    # Loop over masked_array_list
    for i, ax in enumerate(axs.flat):
        if i < num_plots:
            ax.imshow(masked_array_list[i], cmap='jet')
            ax.axis('off')
        else:
            ax.axis('off')
    
    # Plot overall plot
    plt.tight_layout()  
    plt.show()


def tiffs_to_foliumMap(tif_path, pixelated, minTemp, maxTemp):
    '''
    Create a folium map with an overlay of a GeoTIFF image.

    Args:
        tif_path (str): Path to the GeoTIFF file.

    Returns:
        folium.Map: A folium map object with the GeoTIFF overlay.
    '''
    # Import tif 
    lst = rioxarray.open_rasterio(tif_path)
    # Extract values
    data = np.array(lst)[0]

    # Create a masked array
    # data = np.ma.masked_array(data, mask = data < 1)

    # Set the color range from 'jet' colormap
    color_range = np.linspace(0, 1, 256)
    colors_jet_rgba = plt.cm.jet(color_range)
    
    # Convert RGBA to hexadecimal format
    colors_jet_hex = [mcolors.rgb2hex(color) for color in colors_jet_rgba]

    # Define the colormap from blue to red
    cmap = plt.colormaps['jet']
    # Normalize the data between 0 and 1
    norm = colors.Normalize(vmin=minTemp, vmax=maxTemp)
    # Apply the colormap to the normalized data
    colored_data = cmap(norm(data))

    # Set image bounds
    image_bounds = box(*lst.rio.bounds())
    # Extract bounds
    min_x, min_y, max_x, max_y = lst.rio.bounds()
    # Set corner coordinates
    corner_coordinates = [[min_y, min_x], [max_y, max_x]]

    #  Initiate map
    m = folium.Map(
        location=[image_bounds.centroid.y, image_bounds.centroid.x],
        zoom_start=10
     )
    #
    folium.GeoJson(image_bounds.__geo_interface__).add_to(m)

    # Add the OpenStreetMap tile layer with transparent colors
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='CartoDB',
        transparent=True,
    ).add_to(m)

    # Overlay the geotiff over the open street map
    folium.raster_layers.ImageOverlay(
         colored_data,
            bounds=corner_coordinates,
            opacity=0.6,
            interactive=True,
            cross_origin=False,
            pixelated=pixelated,
            zindex=0.2,
            interpolation="nearest"
        ).add_to(m)

    # Create the colormap legend with 'jet' colormap colors
    colormap = cm.LinearColormap(
        colors=colors_jet_hex,
        vmin=minTemp,
        vmax=maxTemp,
        max_labels=15
        )

    colormap.caption = 'Land surface temperature in celsius'
    colormap.add_to(m)
    
    # Display map
    return m


def plot_by_key(key, data ,config):

    if data == 'LSTE': 
        colormap = 'jet'
    else: 
        colormap = None

    onlyfiles = [
        f
        for f in listdir(config['data']['ES_tiffs'])
        if isfile(join(config['data']['ES_tiffs'], f)) and f.endswith('.tif')
        ]

    lst = rioxarray.open_rasterio(
        config['data']['ES_tiffs'] + 
        [f for 
        f in [p for p in onlyfiles if key in p] 
        if data in f and '.tif' in f][0]
        )
    img = np.array(lst)[0]
    plt.imshow(img, cmap=colormap)
    plt.show()


def split_period(period, split_half=True):
    result = []
    format_str = '%Y-%m-%d %H:%M:%S'

    start_date = datetime.strptime(period[0]['start'], format_str)
    end_date = datetime.strptime(period[0]['end'], format_str)
    current_date = start_date

    while current_date < end_date:
        if split_half:
            month_start = current_date.replace(day=1)
            next_month_start = (month_start + timedelta(days=31)).replace(day=1)
            half_duration = (next_month_start - month_start) // 2
            first_half_end = month_start + half_duration
            second_half_start = first_half_end + timedelta(seconds=1)
            second_half_end = next_month_start - timedelta(seconds=1)

            result.append({
                'start': current_date.strftime(format_str),
                'end': first_half_end.strftime(format_str)
            })

            result.append({
                'start': second_half_start.strftime(format_str),
                'end': second_half_end.strftime(format_str)
            })

            current_date = next_month_start
        else:
            month_start = current_date.replace(day=1)
            next_month_start = (month_start + relativedelta(months=1)).replace(day=1)
            month_end = next_month_start - timedelta(seconds=1)

            result.append({
                'start': current_date.strftime(format_str),
                'end': month_end.strftime(format_str)
            })

            current_date = next_month_start

    return result


def invertHeatwave(heatwave):
    
    start_date = date(2022, 6, 1)
    end_date = date(2022, 8, 31)
    date_list = []

    delta = timedelta(days=1)
    current_date = start_date

    while current_date <= end_date:
        date_element = {
            'start': current_date.strftime('%Y-%m-%d 00:00:00'),
            'end': (current_date + delta).strftime('%Y-%m-%d 00:00:00')
        }
        date_list.append(date_element)
        current_date += delta

    inverted_HW = []

    # Loop to invert heatwave
    for dates in date_list:
        if not dateInHeatwave(datetime.strptime(dates['start'],'%Y-%m-%d %H:%M:%S'), heatwave):
            inverted_HW.append(dates)
        elif not dateInHeatwave(datetime.strptime(dates['end'],'%Y-%m-%d %H:%M:%S'), heatwave):
            inverted_HW.append(dates)
    
    return inverted_HW


def calculateTempRange(dwd_data):
    # Extract id for munich central
    id = 3379
    # Reduce the data to munich central
    mCentral = dwd_data[dwd_data['STATIONS_ID'] == id].reset_index(drop=True)
    mCentral.drop('STATIONS_ID', inplace=True,axis=1)
    # Convert the date column to datetime
    mCentral['MESS_DATUM'] = mCentral['MESS_DATUM'].astype('datetime64[ns]')

    # Filter the data for the year 2022
    mCentral_2022 = mCentral[mCentral['MESS_DATUM'].dt.year == 2022]

    minTemp = mCentral_2022.groupby(mCentral_2022['MESS_DATUM'].dt.month)['TT_TU'].min().to_list()
    maxTemp = mCentral_2022.groupby(mCentral_2022['MESS_DATUM'].dt.month)['TT_TU'].max().to_list()

    tempRange = {}

    for i in range(1,13):
        diff = (maxTemp[i-1] - minTemp[i-1])/5
        tempRange[i] = [round(minTemp[i-1]-diff,2), round(maxTemp[i-1]+(diff*5),2)]
    
    return tempRange
    

def meanTiff(period, insideHeatwave, config):

    # Create a Dataframe to check the quality of all relevant tiffs (in heatwave)
    dataQ = dataQualityOverview(period, 0, 25, config)

    # Create DF with relevant tifs for the morning
    morning = dataQ[
        (pd.to_datetime(dataQ['dateTime']).dt.hour >= 5) & 
        (pd.to_datetime(dataQ['dateTime']).dt.hour <= 8) & 
        dataQ['qualityFlag']
        ]

    # Create DF with relevant tifs for the afternoon
    afterNoon = dataQ[
        (pd.to_datetime(dataQ['dateTime']).dt.hour >= 12 ) & 
        (pd.to_datetime(dataQ['dateTime']).dt.hour <= 16) & 
        dataQ['qualityFlag']
        ]
    if insideHeatwave:
        # Create and store mean tiff
        meanAfterNoon, maList = mergeTiffs(morning['orbitNumber'], 'avgMorning_HW.tif', config)
        meanAfterNoon, maList = mergeTiffs(afterNoon['orbitNumber'], 'avgAfterNoon_HW.tif', config)
    else:
        # Create and store mean tiff
        meanAfterNoon, maList = mergeTiffs(morning['orbitNumber'], 'avgMorning_nonHW.tif', config)
        meanAfterNoon, maList = mergeTiffs(afterNoon['orbitNumber'], 'avgAfterNoon_nonHW.tif', config)
