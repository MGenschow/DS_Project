import pandas as pd
import pickle
import numpy as np 
import geopandas as gpd
from glob import glob
import rasterio
from rasterio import features
import rioxarray
import shapely.geometry as sg
from tqdm import tqdm
from shapely.geometry import box
import re
from PIL import Image

# %%
# Load config file and get orthophoto data path
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
orthophoto_dir = config['data']['orthophotos']
predictions_dir = config['data']['segmentation_output']
nutzungsdaten_dir = config['data']['nutzungsdaten']

# %% [markdown]
# ### Helper functions

# %%
# Function to get all image paths
def get_file_paths():
    """
    Retrieves the file paths of image and mask files in the 'Patched' directory.

    Returns:
        dict: A dictionary containing the file paths of image and mask files.
              The dictionary is integer-indexed for easy sampling using an integer index.
              The keys are integers representing the index, and the values are tuples
              containing the image file path and the corresponding mask file path.

    """
    mask_files = glob(orthophoto_dir + '/labeling_subset/raw_annotations/*.png')
    image_files = glob(orthophoto_dir + '/labeling_subset/images/*.tif')

    print(f"Indexing files in orthophotos/labeling_subset... \nFound: \t {len(image_files)} Images \n\t {len(mask_files)} Mask")

    # Get base name of all files and create dict with image and mask file paths
    pattern = '\d+_+\d+_patch_\d{1,2}_\d{1,2}'
    patch_base_names = [re.search(pattern, mask_files[i]).group(0) for i in range(len(mask_files))]
    # The dictionary is integer-indexed to allow the dataset __getitem__ class to sample using an integer idx
    path = orthophoto_dir + '/labeling_subset'
    file_paths = {i:(path+'/images/'+name+'.tif',path+'/raw_annotations/'+name+'.png') for i, name in enumerate(patch_base_names)}
    return file_paths

# %%
def map_rgb2label(mask:np.array):
    """
    Maps an RGB mask to a label mask using a color-to-index mapping.

    Args:
        mask (np.array): An RGB mask represented as a NumPy array.

    Returns:
        np.array: A label mask represented as a NumPy array.
                  The label mask has the same shape as the input mask
                  and contains integer values representing the labels.
    """
    rgb2idx = {(127,246,45):5, (237,110,60):3}
    
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for color, label in rgb2idx.items():
        indices = np.where(np.all(mask == color, axis=-1))
        label_mask[indices] = label
    return label_mask


# %% [markdown]
# ### Read in Nutzungsdaten and Hausrumringe

# %%
nutzungsdaten_df = pickle.load(open(nutzungsdaten_dir + 'nutzungsdaten_relevant.pkl', 'rb'))
nutzungsdaten_df = nutzungsdaten_df[nutzungsdaten_df.Label.isin([4,6])]
nutzungsdaten_df.head(3)

# %%
hausumringe_df = pickle.load(open(config['data']['building_boxes'] + '/processed_building_boxes/building_boxes.pkl', 'rb'))
hausumringe_df['Label'] = 2
hausumringe_df

# %% [markdown]
# ### Define Function to process complete masking

# %%
def create_target_mask(orig_tif_path, mask_path,  nutzungsdaten_df, hausumringe_df, reproject=False):

    ############## Build new transform based on the size of the predictions and the geographic extend of the original image ###########
    # Read in original tif
    orig = rasterio.open(orig_tif_path)
    orig_profile = orig.profile
    orig_bounds = orig.bounds

    new_profile = orig.profile
    new_profile.update(count = 1)


    ################## NUTZUNGSDATEN ###########################
    # Subset nutzungsdaten df to relevant polygons and crop them to extend of the original tif
    bbox = box(*orig.bounds)
    # Subset to all polygons that intersect boudning box
    subset_df = nutzungsdaten_df[nutzungsdaten_df.geometry.intersects(bbox)].copy()
    if len(subset_df) == 0:
        nd_mask = np.zeros((625,625))
    else:
        # Clip the polygons to the bounding box extent
        subset_df['geometry'] = subset_df.geometry.intersection(bbox)

        # Save to tif
        with rasterio.open('temp.tif', 'w+', **new_profile) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom,value) for geom, value in zip(subset_df.geometry, subset_df.Label))

            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)

        # Read in tif again to use its values
        out = rasterio.open('temp.tif')
        nd_mask = out.read(1)

        # delete temporary tif file from disk
        os.remove('temp.tif')

    ##################### HAUSUMRINGE ######################################
    # Subset nutzungsdaten df to relevant polygons and crop them to extend of the original tif
    bbox = box(*orig.bounds)
    # Subset to all polygons that intersect boudning box
    subset_df = hausumringe_df[hausumringe_df.geometry.intersects(bbox)].copy()
    if len(subset_df) == 0:
        hu_mask = np.zeros((625,625))
    else:
        # Clip the polygons to the bounding box extent
        subset_df['geometry'] = subset_df.geometry.intersection(bbox)

        # Save to tif
        with rasterio.open('temp.tif', 'w+', **new_profile) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom,value) for geom, value in zip(subset_df.geometry, subset_df.Label))

            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)

        # Read in tif again to use its values
        out = rasterio.open('temp.tif')
        hu_mask = out.read(1)

        # delete temporary tif file from disk
        os.remove('temp.tif')
    
    ########################## Manual Labeling Annotations
    ann_mask = Image.open(mask_path)
    ann_mask = np.array(ann_mask)
    ann_mask = map_rgb2label(ann_mask)

    ###################### Create final mask
    # Methodology: 
    # - Define everything as impervious surface (background)
    # - add low vegetation
    # - overlay roads as impervious
    # - overly tree annotations
    # - overlay building bounds
    # - put water on top

    final_mask = np.ones((ann_mask.shape[0], ann_mask.shape[1]))
    final_mask[ann_mask == 3] = 3
    final_mask[nd_mask == 6] = 1
    final_mask[ann_mask == 5] = 5
    final_mask[hu_mask == 2] = 2
    final_mask[nd_mask == 4] = 4

    out_name = mask_path.split('/')[-1].split('.')[0] + '.tif'
    out_path = orthophoto_dir + '/labeling_subset/final_masks/' + out_name
    
    # Write final prediction to tif
    with rasterio.open(out_path, 'w', **new_profile) as dst:
        dst.write(final_mask, 1)

    if reproject:
        # Read in again, reproject and save again
        dst = rioxarray.open_rasterio(out_path)
        dst = dst.rio.reproject('EPSG:4326')
        dst.rio.to_raster(out_path)


# %% [markdown]
# ### Process all masks

# %%
print('Processing all masks...')
paths = get_file_paths()
for i in tqdm(range(len(paths))):
    orig_tif_path = paths[i][0]
    mask_path = paths[i][1]

    create_target_mask(orig_tif_path, mask_path, nutzungsdaten_df,hausumringe_df, reproject=False)



# %%
