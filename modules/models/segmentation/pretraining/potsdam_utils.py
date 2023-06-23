# Basic Imports
import glob
import re
import os
import numpy as np
import random
from tqdm import tqdm

# Torch Imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

# Image utils imports
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# Load config file and get potsdam data path
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path = config['data']['potsdam']

################################################################################################
###################################### File Path Utils and Train/Test Split ####################
################################################################################################

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
    image_files = glob.glob(path+'/Patched/*image*.tif')
    mask_files = glob.glob(path+'/Patched/*mask*.tif') 

    print(f"Indexing files in potsdam/Patched... \nFound: \t {len(image_files)} Images \n\t {len(mask_files)} Mask")

    # Get base name of all files and create dict with image and mask file paths
    pattern = 'top_potsdam_\d{1,2}_\d{1,2}_patch_\d{1}_\d{1}'
    patch_base_names = [re.search(pattern, image_files[i]).group(0) for i in range(len(image_files))]
    # Get rid of image 'top_potsdam_4_12 because mask RGB mapping has errors and does not adhere to the mapping convention
    patch_base_names = [name for name in patch_base_names if 'potsdam_4_12_' not in name]
    # The dictionary is integer-indexed to allow the dataset __getitem__ class to sample using an integer idx
    file_paths = {i:(path+'/Patched/'+name+'_image.tif',path+'/Patched/'+name+'_mask.tif') for i, name in enumerate(patch_base_names)}
    return file_paths

def train_test_split(file_paths:dict, test_size:float=0.2):
    """
    Splits a dictionary of file paths into training and test sets.

    Args:
        file_paths (dict): A dictionary containing the file paths of image and mask files.
                           The keys are integers representing the index, and the values are tuples
                           containing the image file path and the corresponding mask file path.
        test_size (float, optional): The proportion of the dataset to include in the test set.
                                     Default is 0.2 (20% of the dataset).

    Returns:
        tuple: A tuple containing two dictionaries representing the training and test sets.
               Each dictionary is integer-indexed for easy sampling using an integer index.
               The keys are integers representing the index, and the values are tuples
               containing the image file path and the corresponding mask file path.
    """
    from sklearn.model_selection import train_test_split

    train_keys, test_keys = train_test_split(list(file_paths.keys()), test_size=test_size)
    train_dict = {i:file_paths[key] for i,key in enumerate(train_keys)}
    test_dict = {i:file_paths[key] for i,key in enumerate(test_keys)}
    print(f"Length of all files: {len(file_paths)}")
    print(f"Length of train ({len(train_dict)}) and test ({len(test_dict)}): {len(train_dict)+len(test_dict)}")
    return train_dict, test_dict

########################################################################################
###################################### Mask Tranform Utils #############################
########################################################################################


# Get label RGB Mpaaings in all directions
# Define official RGB to Label mappings
RGB_classes = [
       (255, 255, 255), # impervious surfaces
       (0,  0, 255), # building
       (0, 255, 255), # low vegetation
       (0,  255,  0), # tree
       (255, 255, 0), # car
       (255, 0, 0)] # clutter/background
Label_classes = [
        "impervious surfaces",
        "building",
        "low vegetation",
        "tree",
        "car",
        "clutter/background"]

idx2label = {key: value for key, value in enumerate(Label_classes)}

# Create a dictionary to translate a mask to a rgb tensor
idx2rgb = {key: value for key, value in enumerate(RGB_classes)}
rgb2idx = {v: k for k, v in idx2rgb.items()}

# Dict to map from label to rgb
rgb2label = dict(zip(Label_classes, RGB_classes))

# Functions to translate labels masks back and forth
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
    
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for color, label in rgb2idx.items():
        indices = np.where(np.all(mask == color, axis=-1))
        label_mask[indices] = label
    return label_mask

def map_label2rgb(label_mask:np.array):
    """
    Maps a label mask to an RGB mask using an index-to-color mapping.

    Args:
        label_mask (np.array): A label mask represented as a NumPy array.
                               The label mask contains integer values representing the labels.

    Returns:
        np.array: An RGB mask represented as a NumPy array.
                  The RGB mask has the same shape as the input label mask and contains
                  color values represented as (R, G, B) tuples.
    
    """
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype='uint8')
    for i in range(label_mask.shape[0]):
        for j in range(label_mask.shape[1]):
            try:
                rgb[i,j,:] = idx2rgb[label_mask[i,j].item()]
            except:
                rgb[i,j,:] = [0,0,0]
    return rgb

################################################################################################
############################## Custom Dataset Class and DataLoader #############################
################################################################################################
class PotsdamDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_filepath = self.file_paths[idx][0]
        mask_filepath = self.file_paths[idx][1]

        image = np.array(Image.open(image_filepath))
        mask = np.array(Image.open(mask_filepath))
        # Convert RGB mask to label mask
        label_mask = map_rgb2label(mask)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_mask)
            transformed_image = transformed['image']
            transformed_label_mask = transformed['mask']
            return transformed_image, transformed_label_mask
        else:
            return image, label_mask

# Define transforms to be used in the Training
train_transform = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.25),
        #A.RandomCrop(500, 500),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ],
)

test_transform = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ],
)

def get_potsdam_loaders(batch_size=2):
    file_paths = get_file_paths()
    train_dict, test_dict = train_test_split(file_paths, test_size=0.2)
    BATCH_SIZE = 2
    train_loader = DataLoader(PotsdamDataset(train_dict, transform=train_transform), 
                            batch_size = BATCH_SIZE, 
                            num_workers = 2)
    test_loader = DataLoader(PotsdamDataset(test_dict, transform=test_transform),
                            batch_size = BATCH_SIZE, 
                            num_workers = 2)
    print(f"Length of train loader: {len(train_loader)}; Length of test loader: {len(test_loader)} with batch size {BATCH_SIZE}")

    return train_loader, test_loader

################################################################################################
####################################### Plotting Functions #####################################
################################################################################################
def denormalize_image(img:np.array):
    """
    Denormalizes an image array that has been normalized using mean and standard deviation.

    Args:
        img (np.array): An image array that has been normalized.

    Returns:
        np.array: The denormalized image array, restored to the original scale.
    """
    
    # Define the mean and standard deviation values used for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Expand dimensions of mean and std arrays to match the shape of the image array
    mean = np.expand_dims(np.expand_dims(mean, axis=1), axis=1)
    std = np.expand_dims(np.expand_dims(std, axis=1), axis=1)

    # Convert the normalized image array back to the original scale
    original_image_array = img * std + mean
    original_image_array = np.clip(original_image_array, 0, 1)  # Clip values between 0 and 1

    # Transpose the shape of the image array to (1000, 1000, 3) for plotting
    #original_image_array = np.transpose(original_image_array, (1, 2, 0))
    return original_image_array

def plot_example(img:Image, mask:Image, title:str='', plot_legend:bool=True):
    """
    Plots an example image and its corresponding mask side by side.

    Args:
        img (PIL.Image): The image to be plotted.
        mask (PIL.Image): The mask to be plotted.
        title (str, optional): The title of the plot. Default is an empty string.
        plot_legend (bool, optional): Specifies whether to plot the legend. Default is True.

    Returns:
        None.
    """
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img)
    ax[1].imshow(mask)

    # Set figure title
    plt.suptitle(title)
    fig.tight_layout()

    if plot_legend:
        # Extract colors and labels from the dictionary
        colors = [(r / 255, g / 255, b / 255, 1) for r, g, b in idx2rgb.values()]
        labels = [str(idx2label[key]) for key in idx2rgb]

        # Create a legend using the extracted colors and labels
        patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
        fig.legend(patches, labels, loc=7)
        fig.subplots_adjust(right=0.75)
    # Remove axes ticks
    [ax[i].get_xaxis().set_ticks([]) for i in range(ax.shape[0])]
    [ax[i].get_yaxis().set_ticks([]) for i in range(ax.shape[0])]

    plt.show()