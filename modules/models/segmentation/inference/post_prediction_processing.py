# %%
import pandas as pd
import pickle
import numpy as np 
import geopandas as gpd
from glob import glob
%cd ../pretraining/
#from loveda_utils import map_label2rgb, idx2rgb, idx2label
from munich_utils import *
%cd ../inference
import matplotlib.pyplot as plt
#import torch
#from torchvision.transforms import ToPILImage
import rasterio
import rioxarray
import folium
from rasterio import features
import shapely.geometry as sg
from tqdm import tqdm
from shapely.geometry import box
import torch.nn as nn

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
# ### Create masked tif out of predictions

# %%
# %%
def create_mask_tif(prediction, orig_tif_path, nutzungsdaten_df, out_name, reproject=False):

    ############## Build new transform based on the size of the predictions and the geographic extend of the original image ###########
    # Read in original tif
    orig = rasterio.open(orig_tif_path)
    orig_profile = orig.profile
    orig_bounds = orig.bounds

    # Calculate spatial extend of original tif
    bounds_width = orig_bounds.right - orig_bounds.left
    bounds_height =  orig_bounds.bottom - orig_bounds.top

    # Calculate new resolution to correct transform affine in profile of new tif
    new_width = prediction.shape[0]
    new_height = prediction.shape[1]

    scale_x = bounds_width / new_width
    scale_y = bounds_height / new_height


    # Update the affine transformation parameters, keep b, c, d and f of original transform, replace a nd e with new resolution
    new_transform = rasterio.Affine(scale_x, orig.transform.b, orig.transform.c,
                                orig.transform.d, scale_y, orig.transform.f)

    # Create a new array with the desired size (511, 512) and populate it with the values from the original array
    # Write the new array to a GeoTIFF file using the geotag data and the new Transform object
    new_profile = orig.profile
    new_profile.update(transform=new_transform, width=new_width, height=new_height, count = 1)

    ############## Transform nutzungsdaten shapefile to geotif with corresponding geographic information
    # Subset nutzungsdaten df to relevant polygons and crop them to extend of the original tif
    bbox = box(*orig.bounds)
    # Subset to all polygons that intersect boudning box
    subset_df = nutzungsdaten_df[nutzungsdaten_df.geometry.intersects(bbox)].copy()
    if len(subset_df) == 0:
        nd_mask = np.zeros((512,512))
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

    ############### Recode Predictions to facilitate nutzungsdaten from Opendata Bayern ################
    # Methodology: 
    # - Define everything as impervious surface (background)
    # - add low vegetation predictions
    # - overlay roads from Nutzungsdaten
    # - add tree predictions
    # - add building predictions
    # - overlay water from Nutzungsdaten

    final_mask = np.ones((prediction.shape[0], prediction.shape[1]))
    final_mask[prediction == 3] = 3
    final_mask[nd_mask == 6] = 6
    final_mask[prediction == 5] = 5
    final_mask[prediction == 2] = 2
    final_mask[nd_mask == 4] = 4
    final_mask[nd_mask == 7] = 1


    # Write final prediction to tif
    with rasterio.open(out_name, 'w', **new_profile) as dst:
        dst.write(final_mask, 1)

    if reproject:
        # Read in again, reproject and save again
        dst = rioxarray.open_rasterio(out_name)
        dst = dst.rio.reproject('EPSG:4326')
        dst.rio.to_raster(out_name)

# %%
def process_all_patches():
    # Device setup
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    print(f"Using {DEVICE} as DEVICE")

    # Load the model
    model = torch.load(config['data']['model_weights']+'/finetuning/model.pt', map_location=torch.device('cpu'))
    model = model.to(DEVICE)
    model.eval()

    nutzungsdaten_df = pickle.load(open(nutzungsdaten_dir + 'nutzungsdaten_relevant.pkl', 'rb'))
    test_loader = get_munich_test_loader(batch_size=16)
    j = 0
    # Outer Loop
    for data, file_paths in tqdm(test_loader):
        # Get model predictions and save them
        with torch.no_grad():
            data = data.to(DEVICE)
            logits = model(data)['out']
            probs = nn.Softmax(logits).dim
            preds = torch.argmax(probs, dim = 1)
        
        # Inner loop: Create mask tifs out of predictions
        for i in range(data.shape[0]):
            # Get tile name and path and prediction
            file_path = file_paths[i]
            pred = preds[i,:,:].cpu()
            tile_name = file_path.split('/')[-1].split('.')[0]
            out_dir = predictions_dir + tile_name + '_mask.tif'
            # Create final mask tif
            create_mask_tif(pred, file_path, nutzungsdaten_df, out_dir)
            #print(f"Saved tile {tile_name} to {out_dir}")

# %%
# Run the function
process_all_patches()

# %% [markdown]
# ### Create Shapefiles of Predictions

# %%
def get_class_polygons(mask_path):
    # Define how classes will be named
    class_labels = {0:'ignore', 1:'impervious', 2:'building', 3:'low vegetation', 4:'water', 5:'trees', 6:'road', 7:'train', 255:'ignore'}

    src = rasterio.open(mask_path)
    transform = src.transform
    data = src.read()

    # initialize empty GeoDataFrame to store all polygons
    gdf = gpd.GeoDataFrame(columns = ['label', 'geometry'])
    for class_index in np.unique(data):
        mask = data == class_index
        # Create a generator of polygon geometries
        shapes = features.shapes(mask.astype(rasterio.uint8), transform=transform)
        # Convert the geometries to Shapely polygons
        polygons = [sg.shape(shape) for shape, value in shapes if value == 1] # if condition necessary to get only positive mask polygons, not negative

        multipolygon = sg.MultiPolygon(polygons)
        data_to_append = {'label':class_labels[class_index], 'geometry':multipolygon}
        gdf = gpd.GeoDataFrame(pd.concat([gdf, pd.DataFrame([data_to_append])]))
        
    return gdf


# %%
all_masks = glob(predictions_dir+ '*.tif')
all_gdfs = []
for file in tqdm(all_masks, leave = False):
    # Create Geopandas df with all shapes
    gdf = get_class_polygons(file)
    all_gdfs.append(gdf)
gdf = gpd.GeoDataFrame(pd.concat(all_gdfs))
gdf.crs = 'EPSG:25832'


with open(config['data']['data'] + '/uhi_model/raw/input.pkl', 'wb') as file:         
    pickle.dump(gdf, file)
# %% 
