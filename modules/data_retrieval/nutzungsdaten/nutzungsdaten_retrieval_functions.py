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

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
nutzungsdaten_path = config['data']['nutzungsdaten']

# Define all URLs that are necessary for bounding box
urls = [
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09162.zip", 
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09184.zip",
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09175.zip",
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09177.zip",
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09178.zip",
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09186.zip", 
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09174.zip",
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09179.zip", 
    "https://download1.bayernwolke.de/a/tn/lkr/tn_09188.zip", 
]


def download_and_unzip(urls, folder_path):
    """
    Download and unzip files from the given URLs into the specified folder.

    Args:
        urls (List[str]): A list of URLs to download and unzip.
        folder_path (str): The path to the folder where the files will be saved.

    Returns:
        None
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    for url in urls:
        # Extract the filename from the URL
        filename = url.split("/")[-1]
        file_path = os.path.join(folder_path, filename)

        # Download the file
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)

        #Create a folder with the same name as the zip file
        zip_folder = os.path.join(folder_path, os.path.splitext(filename)[0])
        os.makedirs(zip_folder, exist_ok=True)

        # Unzip the file into the created folder
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(zip_folder)

        # Remove the downloaded zip file
        os.remove(file_path)



def create_complete_dataset():
    """
    Create a complete dataset by reading multiple shapefiles, concatenating them into one Geopandas DataFrame,
    and saving the result as a pickle file.

    Returns:
        None
    """
    dfs = []
    for file in tqdm(glob(nutzungsdaten_path + '*/'), leave=False):
        dfs.append(gpd.read_file(file))
    # Concatenate all the files into one dataframe
    df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
    # Convert the geometry column to the correct CRS
    df['geometry_4326'] = df['geometry'].to_crs(epsg=4326)
    
    # Save to pickle
    with open(nutzungsdaten_path + 'nutzungsdaten_complete.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def subset_nutzungsdaten_df():
    df = pickle.load(open(nutzungsdaten_path + 'nutzungsdaten_complete.pkl', 'rb'))

    # Relabel to relevant labels for our case
    df['Label'] = np.nan
    df['Label'] = np.where(df.nutzart.isin(['Fließgewässer', 'Stehendes Gewässer', 'Hafenbecken', 'Schiffsverkehr']), 4, df['Label'])
    #df['Label'] = np.where(df.nutzart.isin(['Wohnbaufläche']), 'Wohnbaufläche', df['Label'])
    df['Label'] = np.where(df.nutzart.isin(['Straßenverkehr', 'Weg']), 6, df['Label'])
    df['Label'] = np.where(df.nutzart.isin(['Bahnverkehr', 'Flugverkehr']), 7, df['Label'])


    # Subset to relevant labels and columns
    df = df[df.Label.isna() == False]
    df = df[['geometry', 'geometry_4326', 'Label']]
    
    # Subset to relevant bounding box
    df = df[df.geometry_4326.intersects(box(*config['bboxes']['munich']))]
    
    df = df.reset_index(drop=True)

    # Save to pickle
    with open(nutzungsdaten_path + 'nutzungsdaten_relevant.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Run functions
#download_and_unzip(urls, nutzungsdaten_path)
#create_complete_dataset()
subset_nutzungsdaten_df()