# %%
from osgeo import gdal
from glob import glob
from tqdm import tqdm

# %%
# Read config and get root directory
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
root_dir = config['data']['orthophotos']

# %%
# Get path to all tiles stored in the "raw_tiles" folder
all_tiles = glob(root_dir + "/raw_tiles/*.tif")
all_tiles = [
    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/orthophotos/raw_tiles/32692_5337.tif',
#    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/orthophotos/raw_tiles/32692_5336.tif',
#    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/orthophotos/raw_tiles/32692_5335.tif'
]

# %%
def slice_image(input_path, output_dir,  num_subtiles):
    """
    Slice the input image into multiple subsets and save them as separate files.

    Args:
        input_path (str): The path to the input image file.
        output_dir (str): The directory where the subset files will be saved.
        num_subtiles (int): The number of subsets to create in EACH dimension. If this is set to 10, the result will be 10x10 = 100 subtiles

    Returns:
        None
    """
    # Prepare filenames for saving
    tile_name = input_path.split('/')[-1].split('.')[0]
    save_path = output_dir + "/" + tile_name + "_patch_"

    #print(f"Processing image {tile_name}")

    # Read in  tif as GDAL dataset
    dataset = gdal.Open(input_path)

    # Get geotransform
    gt = dataset.GetGeoTransform()
    # Get upper left coordinate of whole image and resolution
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]

    # Calculate length of axes
    xlen = res * dataset.RasterXSize
    ylen = res * dataset.RasterYSize

    # calculate size of one subset
    xsize = xlen/num_subtiles
    ysize = ylen/num_subtiles

    # Calculate x-coordinates where a split takes place
    xsteps = [xmin + xsize * i for i in range(num_subtiles+1)]
    ysteps = [ymax - ysize * i for i in range(num_subtiles+1)]

    # convolute over image and save subset
    for i in range(num_subtiles):
        for j in range(num_subtiles):
            # calculate corner coordinates of subset
            xmin = xsteps[i]
            xmax = xsteps[i+1]
            ymax = ysteps[j]
            ymin = ysteps[j+1]

            # Save patch to disk
            gdal.Warp(save_path + str(i) + "_" + str(j)+".tif", 
                        dataset, 
                        outputBounds = (xmin, ymin, xmax, ymax), dstNodata = None)


# %%
# Loop over all tiles and create corresponding subtiles
patch_size = 4
print(f"Processing {len(all_tiles)} tiles with patch size {patch_size}, i.e. {patch_size**2} subtiles for each image ...")
for tile in tqdm(all_tiles):
    slice_image(tile, 
                output_dir="/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/orthophotos/patched", 
                num_subtiles=patch_size)
    
    
# %%
