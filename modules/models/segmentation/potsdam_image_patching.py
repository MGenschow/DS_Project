# Imports
import glob
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
import re

### Dataset Slicing
#- The Potsdam dataset consists only of 38 (image, mask) pairs with size 6000x6000. 
#- The resolution per pixel is 0.05 meters, resulting in a 300x300m size of one image
#- Below I slice the image to patches of size 2000x2000 such that each image represents a 100x100m quadrant
#- This is done to create more training examples and at the same time to reduce the dimension of the data passed through the neural network downstream

# Read path from config file
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path = config['data']['potsdam']



# Get the list of files
mask_files = glob.glob(path+'/*label.tif')
image_files = glob.glob(path+'/2_Ortho_RGB/*RGB.tif') 
print(f"Found: \n {len(mask_files)} mask files \n {len(image_files)} image files")

# The tile name can be extracted from each file. Below I create a dictionary to match tile name to a file name for images and masks
pattern = 'top_potsdam_\d{1,2}_\d{1,2}'
mask_files_dict = {re.search(pattern, file).group(0):file for file in mask_files}
image_files_dict = {re.search(pattern, file).group(0):file for file in image_files}
# The length of the dicts and their keys should be identical, since each tile consists of a (image, mask) pair
# Check identity condition
assert len(mask_files_dict) == len(image_files_dict)
assert mask_files_dict.keys() == image_files_dict.keys()


# Function to create patches
def create_patches(tile_name:str, patch_size:int = 2000):
    print(f"Processing image {tile_name}")
    img_path = image_files_dict[tile_name]
    mask_path = mask_files_dict[tile_name]

    # Read in the image and mask and transform to tensor
    image = Image.open(img_path)
    mask = Image.open(mask_path)

    image = ToTensor()(image)
    mask = ToTensor()(mask)

    # Calculate number of patches in each dimension
    num_patches_height = image.shape[1] // patch_size
    num_patches_width = image.shape[2] // patch_size

    # Loop over the patches and crop image and mask accordingly
    for h in range(num_patches_height):
        for w in range(num_patches_width):
            patch_image = image[:, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
            patch_target = mask[:, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]

            # retransform to PIL image
            patch_image = ToPILImage()(patch_image)
            patch_target = ToPILImage()(patch_target)

            # Save the patches
            if not os.path.exists(path+"/Patched/"):
                os.makedirs(path+"/Patched/")
            patch_image.save(path+f"/Patched/{key}_patch_{h}_{w}_image.tif")
            patch_target.save(path+f"/Patched/{key}_patch_{h}_{w}_mask.tif")
            print(f"\t Saved patch {h}_{w}")

# Create patches for all tiles       
patch_size = 2000
for key in image_files_dict.keys():
    create_patches(key, patch_size)