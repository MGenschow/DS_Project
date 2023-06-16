# %%
import os
import urllib.request
import zipfile
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import pandas as pd
import pickle
import numpy as np
from shapely.geometry import box
import pandas as pd
import xml.etree.ElementTree as ET 
from tqdm import tqdm

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
city_3d_model_path = config['data']['city_3d_model']

# %%
# Define all URLs that are necessary for bounding box
urls = [
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09162.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09184.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09175.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09177.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09178.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09186.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09174.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09179.meta4', 
    'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09188.meta4'
]


def download_file_from_urls(urls, folder_path):
    """
    Downloads files from a list of URLs and saves them to the specified folder path.

    Args:
        urls (list): A list of URLs of the files to be downloaded.
        folder_path (str): The path to the folder where the files will be saved.

    Returns:
        None
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    for url in tqdm(urls):
        # Extract the filename from the URL
        filename = url.split("/")[-1]
        file_path = os.path.join(folder_path, filename)

        # Download the file
        #print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)

# %%
def parse_meta4_file(file):
    """
    Parses a meta4 file and extracts the attributes of each file element.

    Args:
        file (str): The path to the meta4 file.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted attributes of each file element.

    Raises:
        FileNotFoundError: If the specified file does not exist.
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
    for file_elem in tqdm(root.iter('{urn:ietf:params:xml:ns:metalink}file')):
        file_dict = {}
        file_dict['tile_name'] = file_elem.get('name')
        file_dict['size'] = int(file_elem.find('{urn:ietf:params:xml:ns:metalink}size').text)
        file_dict['hash_type'] = file_elem.find('{urn:ietf:params:xml:ns:metalink}hash').get('type')
        file_dict['hash_value'] = file_elem.find('{urn:ietf:params:xml:ns:metalink}hash').text
        urls = file_elem.findall('{urn:ietf:params:xml:ns:metalink}url')
        file_dict['url1'] = urls[0].text if len(urls) > 0 else None
        file_dict['url2'] = urls[1].text if len(urls) > 1 else None
        file_list.append(file_dict)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(file_list)
    return df

# %%
# Download .meta4 files
download_file_from_urls(urls, city_3d_model_path + '/metafiles')
# create df with all .meta4 information
dfs = []
for file in glob(city_3d_model_path + 'metafiles/*.meta4'):
    dfs.append(parse_meta4_file(file))
df = pd.concat(dfs)

# Download all gml files
download_file_from_urls(df.url1, city_3d_model_path + '/raw_gml')


# %%
