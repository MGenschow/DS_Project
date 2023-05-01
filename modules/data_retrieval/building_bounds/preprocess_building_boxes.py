import pandas as pd
import geopandas as gpd
import os
import urllib
import zipfile
import pickle
import shutil
from shapely.geometry import box

import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class BuildingBoxDownloader():
    """
    A class to download and process building box data for a specified bounding box from the Bavarian Open Data Portal.
    
    Attributes:
    -----------
    bbox : tuple
        A tuple containing the coordinates of the bounding box in the format (minx, miny, maxx, maxy).
    data_dir : str
        A string containing the directory path to the root folder for the building box data.
    """
    def __init__(self, bbox, data_dir=config['data']['building_boxes']):
        self.bbox = bbox
        self.data_dir = data_dir

    def prepare_relevant_building_boxes(self):
        """
        Wrapper function to perform all steps needed to find and download the raw building box files from the 
        OpenData portal of Bavaria and process them by subsetting to only relevant areas as specified by the class 
        configuration. 

        Args:
        None

        Returns:
        None
        """
        self._create_shapefile_for_regierungsbezirke()
        urls = self._get_relevant_urls()
        self._download_building_boxes(urls)
        self._process_building_boxes(urls)
    

    # function to download and prepare data for regireungsbezirke
    def _create_shapefile_for_regierungsbezirke(self):
        """
        Downloads and processes shapefiles of administrative regions in Bavaria from a public open data portal. 
        Creates a subset of the shapefiles containing only the Regierungsbezirk regions and saves the subset as 
        a pickle file for future use. If the subset already exists, then the function does not perform any 
        operations and simply prints a message to indicate that the file already exists.

        Returns:
        None
        """

        dir = os.path.join(self.data_dir, 'bezirke_shapefiles')
        # Check if shapefile already exists
        if not os.path.exists(os.path.join(dir, 'bezirke.pkl')):
            # Download zip files from open data portal
            zip_path = os.path.join(dir, 'verwaltungsgebiete.zip')
            if not os.path.exists(dir):
                os.makedirs(dir)
            url = 'https://geodaten.bayern.de/odd/m/4/verwaltung/alkis-verwaltung.zip'
            urllib.request.urlretrieve(url, zip_path)

            # Extract files from zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(dir, 'verwaltungsgebiete'))
            
            # Delete zip files
            os.remove(zip_path)

            # Read shape files and subset to regierungsbezirke
            df = gpd.read_file(os.path.join(dir, 'verwaltungsgebiete', 'ALKIS-Vereinfacht'))
            bezirke = df[df.art == 'Regierungsbezirk']

            bezirke['geometry_4326'] = bezirke['geometry'].to_crs(epsg=4326)
            pickle.dump(bezirke, open(os.path.join(dir, 'bezirke.pkl'), 'wb'))

            # Delete initial shapefile
            shutil.rmtree(os.path.join(dir, 'verwaltungsgebiete'))
        else:
            print('Shapefile for regierungsbezirke already exists.')
    
    def _get_relevant_urls(self):
        """
        Returns a dictionary of URLs to download relevant Hausumringe data for the Regierungsbezirke regions
        specified by the input bounding box.

        Args:
        self: An instance of the class that contains information about the bounding box and directory paths.

        Returns:
        A dictionary with the names of the relevant regierungsbezirke regions as keys and URLs as values.
        """

        # Read in shapefile information
        dir = os.path.join(self.data_dir, 'bezirke_shapefiles')
        bezirke = pickle.load(open(os.path.join(dir, 'bezirke.pkl'), 'rb'))
        # Subset based on bounding box to only relevant regierungsbezirke
        subset = bezirke[bezirke.geometry_4326.intersects(self.bbox)]

        # Construct urls
        base_url = 'https://geodaten.bayern.de/odd/m/3/daten/hausumringe/bezirk/data/20221130_'
        relevant_urls = {}
        for i, row in subset.iterrows():
            name = row['name'].split()[1]
            relevant_urls[name] = base_url+ row['ags'] + '_' + name + '_Hausumringe.zip'
        return relevant_urls
    
    def _download_building_boxes(self, urls:dict):
        """
        Downloads raw building box files for the Regierungsbezirk regions specified in the input dictionary of URLs. 
        If a file already exists, it is not downloaded again. The downloaded files are saved to the directory 
        specified in the class configuration file.

        Args:
        urls: A dictionary containing the names of the relevant regierungsbezirke regions as keys and 
        URLs to download the raw building box files as values.

        Returns:
        None
        """
        dir = dir = os.path.join(config['data']['building_boxes'], 'raw_building_box_files')
        if not os.path.exists(dir):
            os.makedirs(dir) 

        for name, url in urls.items():
            path = os.path.join(dir, name)
            if not os.path.exists(path):
                print(f'Downloading raw building boxes file for {name}... This may take some time.')
                urllib.request.urlretrieve(url, path+'.zip')
                with zipfile.ZipFile(path+'.zip', 'r') as zip_ref:
                    zip_ref.extractall(path)
                os.remove(path+'.zip')
            else:
                print(f'Raw building boxes file for {name} already exists.')

    def _process_building_boxes(self, urls:dict):
        """
        Processes the raw building box files for the Regierungsbezirk regions specified in the input dictionary of URLs. 
        The resulting GeoDataFrame is reprojected to EPSG 4326 and subsets to the bounding box specified in the 
        class configuration file. The processed building boxes are saved to the directory specified in the 
        class configuration file.

        Args:
        urls: A dictionary containing the names of the relevant regierungsbezirke regions as keys and 
        URLs to download the raw building box files as values.

        Returns:
        None
        """
        print('Processing building boxes...')
        # Read in relevant shapefiles
        dir = os.path.join(self.data_dir, 'raw_building_box_files')
        building_boxes = gpd.GeoDataFrame()
        for name in urls.keys():
            data = gpd.read_file(os.path.join(dir, name))
            building_boxes = pd.concat([building_boxes, data])
        # Reproject to epsg 4326
        building_boxes['geometry_4326'] = building_boxes['geometry'].to_crs(epsg=4326)
        # Subset to relevant bounding box
        building_boxes = building_boxes[building_boxes.geometry_4326.intersects(self.bbox)]
        print(f'Found {len(building_boxes)} building boxes in the given bounding box.')

        # Save processed building boxes
        dir = os.path.join(self.data_dir, 'processed_building_boxes')
        if not os.path.exists(dir):
            os.makedirs(dir)
            pickle.dump(building_boxes, open(os.path.join(dir, 'building_boxes.pkl'), 'wb'))
        

