# %%
import pandas as pd 
import geopandas as gpd 
import numpy as np 
import pickle
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
import folium
from shapely.geometry import box, Polygon
import rioxarray
from torchvision.transforms import ToPILImage
import os
from tqdm import tqdm

# %%
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
orthophoto_dir = config['data']['orthophotos']
grid_dir = config['data']['data'] + '/uhi_model/'

# %% [markdown]
# #### Read in Grid File

# %%
with open(grid_dir + 'final_250_e.pkl', 'rb') as f:
    grid = pickle.load(f)
grid = grid[['geometry', 'id']]
grid.head(2)

# %% [markdown]
# #### Read in Orthophoto Index

# %%
with open(orthophoto_dir + '/metalink_files/complete_index.pkl', 'rb') as f:
    ortho_index = pickle.load(f)
ortho_index = gpd.GeoDataFrame(ortho_index)
ortho_index.set_geometry('polygon_4326', inplace=True)

# %% [markdown]
# #### Identify Relevant tiles

# %%
def get_relevant_tiles(bbox):
    return list(ortho_index[ortho_index.polygon_4326.intersects(bbox)].tile_name)

# %%
grid['relevant_tiles'] = grid.geometry.map(get_relevant_tiles)
grid['len'] = grid.relevant_tiles.map(len)
grid.len.value_counts(normalize = True)*100

# %% [markdown]
# #### Merging, Cropping, Resizing and Saving

# %%
def crop_grid_photo(row):
    # Read in all relevant tiles
    tiles = [rasterio.open(orthophoto_dir + '/raw_tiles/' + row.relevant_tiles[i]) for i in range(row.len)]
    # Merge tiles
    arr, out_trans = merge(tiles)

    # Metadata from first file
    out_meta = tiles[0].meta.copy()

    # Update the metadata
    out_meta.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "transform": out_trans,
        # You may also need to update the "count" if merging layers with different number of bands
        "count": arr.shape[0]
    })

    # Write the merged array to the new raster file
    with rasterio.open('test.tif', 'w', **out_meta) as dest:
        dest.write(arr)

    # Reproject to EPSG:4326

    # Read in again, reproject and save again
    dst = rioxarray.open_rasterio('test.tif')
    dst = dst.rio.reproject('EPSG:4326')
    dst.rio.to_raster('test.tif')

    # Crop image
    grid_geom = row.geometry
    with rasterio.open('test.tif') as src: # replace with your path
                out_image, out_transform = mask(src, [grid_geom], crop=True)
                out_meta = src.meta.copy()

    # Resize and convert to PIL
    img = ToPILImage()(out_image.transpose(1,2,0)).resize((200,200))
    img.save(f"{orthophoto_dir}/jpeg_test/{row.id}.jpg")

    # Remove tif
    os.remove('test.tif')

# %%
#crop_grid_photo(grid.iloc[40,:])

# %%
for i in tqdm(range(4190,len(grid))):
    row = grid.iloc[i,:]
    crop_grid_photo(row)


