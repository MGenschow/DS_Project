import pandas as pd
import xml.etree.ElementTree as ET 
import urllib
import re
import os
import geopandas as gpd
import rasterio
import rasterio.warp
from shapely.geometry import Polygon
import yaml


# Load configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



class orthophoto_indexer():

    def __init__(self, data_dir=config['data']['orthophotos']):
        self.data_dir = data_dir
        self.regierungsbezirke = {
            "091": "Oberbayern",
            "092": "Niederbayern",
            "093": "Oberpfalz",
            "094": "Oberfranken",
            "095": "Mittelfranken",
            "096": "Unterfranken",
            "097": "Schwaben"
            }

    def download_metalink_per_regierungsbezirk(self, override = False):
        """
        Downloads metalink files for DOP40 files the 7 Regierungsbezirke of Bayern.

        For each Regierungsbezirk, this function retrieves the corresponding metalink file
        from the Bavarian Geodatenzentrum and saves it locally in the `metalink_files` directory.
        Metalink files contain information about the available Digital Orthophoto (DOP) tiles
        for a given geographic region.

        Args:
            override (bool): If True, existing metalink files will be overwritten, otherwise skipped.

        Returns:
            None.
        """
        base_url = 'https://geodaten.bayern.de/odd/a/dop40/meta/metalink/'

        # Check if metalink_files directory exists and create it if not
        metalink_files_dir = os.path.join(self.data_dir, 'metalink_files')
        if not os.path.exists(metalink_files_dir):
            os.mkdir(metalink_files_dir)

        for code, name in self.regierungsbezirke.items():
            url = base_url + code + '.meta4'
            path = os.path.join(metalink_files_dir, f'{name}.meta4')

            if not override and os.path.exists(path):
                print(f'Skipping download of metalink file for Regierungsbezirk {name} to file {path} as it already exists')
                continue
            elif override and os.path.exists(path):
                print(f'Overriding existing metalink file for Regierungsbezirk {name} to file {path}')
                urllib.request.urlretrieve(url, path)
            elif not os.path.exists(path):
                print(f'Downloading metalink file for Regierungsbezirk {name} to file {path}')
                urllib.request.urlretrieve(url, path)
    
    def create_complete_index(self,
                              print_summary=True):
        """
        Creates a complete index of all DOP40 tiles for all 7 Regierungsbezirke of Bayern.

        This function iterates through all metalink files and parses them to extract the
        information about the available DOP40 tiles. The information is stored in a single
        Pandas DataFrame and saved to disk as a .csv file.

        Returns:
            None.
        """
        # Make sure that the metalink_files exist
        self.download_metalink_per_regierungsbezirk(override=False)

        out_path=os.path.join(self.data_dir, 'metalink_files', 'complete_index.pkl')
        
        # Create a list of dictionaries to store the file attributes
        file_list = []

        # Iterate through the metalink files and extract the attributes
        for file in os.listdir(os.path.join(self.data_dir, 'metalink_files')):
            if file.endswith('.meta4'):
                file_list.append(file)
        # Parse all files and create a DataFrame
        complete_index = pd.DataFrame()
        for file in file_list:
            print(f'Parsing file {file}')
            temp_df = self.parse_meta4_file(os.path.join(self.data_dir, 'metalink_files', file))
            complete_index = pd.concat([complete_index, temp_df], ignore_index=True)
        
        # Save file to disk
        complete_index.to_pickle(out_path)
        print(f'Successfully created complete index and saved it to {out_path}')
        

        if print_summary:
            self._print_summary(complete_index)

    def _print_summary(self, df):
        """
        Print a summary of the contents of the complete index dataframe.
        """
        print(f'Total files: {len(df)}')
        print(f'Number of Regierungsbezirke: {len(df.Regierungsbezirk.unique())}')
        print(f'Averge size of files: {round(df.size_mb.mean(),2)} MB')
        print(f'Total size of files: {round(df.size_mb.sum()/1000,2)} GB')
        print(f'Number of files with missing URL: {len(df[df.url1.isna()])}')
        print(f'Number of files & total size per Regierungsbezirk:')
        for regierungsbezirk in df.Regierungsbezirk.unique():
            print(f'        {regierungsbezirk}: {len(df[df.Regierungsbezirk == regierungsbezirk])} -- {round(df[df.Regierungsbezirk == regierungsbezirk].size_mb.sum()/1000,2)} GB')


    def parse_meta4_file(self, file):
        """
        Parse a .meta4 file and return its contents as a Pandas DataFrame.

        Parameters:
        -----------
        meta4_file : str
            The path to the .meta4 file to be parsed.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the file attributes extracted from the .meta4 file.
            The DataFrame has the following columns:
            - name: str, the name of the file.
            - hash_type: str, the type of hash value (e.g., sha-256).
            - hash_value: str, the hash value of the file.
            - url1: str, the first URL to download the file.
            - url2: str, the second URL to download the file.
            - size_mb: float, the size of the file in megabytes.
        """
        #check if file exists
        if not os.path.exists(file):
            raise FileNotFoundError(f'File {file} does not exist.')


        # Parse meta4 file and get root
        tree = ET.parse(file)
        root = tree.getroot()

        # Create a list of dictionaries to store the file attributes
        file_list = []

        # Iterate through the file elements and extract the attributes
        for file_elem in root.iter('{urn:ietf:params:xml:ns:metalink}file'):
            file_dict = {}
            file_dict['tile_name'] = file_elem.get('name')
            file_dict['size'] = int(file_elem.find('{urn:ietf:params:xml:ns:metalink}size').text)
            file_dict['hash_type'] = file_elem.find('{urn:ietf:params:xml:ns:metalink}hash').get('type')
            file_dict['hash_value'] = file_elem.find('{urn:ietf:params:xml:ns:metalink}hash').text
            urls = file_elem.findall('{urn:ietf:params:xml:ns:metalink}url')
            file_dict['url1'] = urls[0].text if len(urls) > 0 else None
            file_dict['url2'] = urls[1].text if len(urls) > 1 else None
            file_dict['polygon_25832'], file_dict['polygon_4326'] = self._get_bounding_boxes_from_filename(file_dict['tile_name'])
            file_list.append(file_dict)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(file_list)

        # Convert the size column from bytes to megabytes
        df['size_mb'] = df['size'] / 1000000
        df = df.drop(columns=['size'])
        
        # Add Regierungsbezirk column
        df['Regierungsbezirk'] = re.search(r'(?:[\/\\])(?P<name>[^\/\\]+)(?=.meta4)', file).group('name')

        return df
    
    def _check_filenames_pattern(self, filename):
        """
        Check if a filename adheres to the pattern 32ddd_dddd.tif with d being a single digit in [0,9].
        
        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the filename adheres to the specified pattern, False otherwise.
        """
        pattern = r'^32\d{3}_\d{4}\.tif$'
        regex = re.compile(pattern)
        if not regex.match(filename):
            return False
        return True

    def _get_bounding_boxes_from_filename(self, filename):
        """
        Extracts bounding boxes from a filename adhering to a specific pattern and returns them as GeoDataFrames.

        Args:
            filename (str): The filename to extract bounding boxes from.

        Returns:
            tuple: A tuple containing two GeoDataFrames representing the bounding boxes. The first one is in EPSG:25832 and
            the second one is in EPSG:4326.

        Raises:
            ValueError: If the filename does not adhere to the pattern '32ddd_dddd.tif'.
        """
        # Check if filename adheres to the pattern 32ddd_dddd.tif with d being a single digit in [0,9] which is needed to extract the bounding box directly from the filename
        if self._check_filenames_pattern(filename):
            # extract bounding box from filename
            filename = filename[2:].split('.')[0]
            xmin = 1000*int(filename.split('_')[0])
            ymin = 1000*int(filename.split('_')[1])
            xmax, ymax = xmin + 1000, ymin + 1000

            xmin, ymin, xmax, ymax
            # Create polygon from bounding box in EPSG_25832
            polygon_25832 = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            #gdf_25832 = gpd.GeoDataFrame({'polygon': [polygon]}, geometry='polygon')

            # Convert bounding box from EPSG:25832 to EPSG:4326
            [xmin, xmax], [ymin, ymax] = rasterio.warp.transform('EPSG:25832', 'EPSG:4326', [xmin, xmax], [ymin, ymax])
            polygon_4326 = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            #gdf_4326 = gpd.GeoDataFrame({'polygon': [polygon]}, geometry='polygon')
            return polygon_25832, polygon_4326
        else:
            raise ValueError(f'Filename {filename} does not adhere to the pattern 32ddd_dddd.tif')
