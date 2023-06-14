# Basic Imports
import glob
import re
import os
import numpy as np
import random
from tqdm import tqdm

# Torch Imports
from torchgeo.datasets import LoveDA
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
config_path = '/home/tu/tu_tu/'+ os.getcwd().split('/')[6] + '/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path = config['data']['LoveDA']

########################################################################################
###################################### Mask Tranform Utils #############################
########################################################################################

# Get label RGB Mpaaings in all directions
# Define official RGB to Label mappings
RGB_classes = [
       (0, 0, 0), # ignore
       (255, 255, 225), # background
       (255,  0, 255), # building
       (255, 0, 0), # road
       (0,  0,  255), # water
       (128, 128, 128), # barren
       (0, 130, 0), # forest
       (255, 200, 0)] # agriculture
Label_classes = [
       "ignore",
       "background",
       "building",
       "road",
       "water",
       "barren",
       "forest",
       "agriculture"]

idx2label = {key: value for key, value in enumerate(Label_classes)}

# Create a dictionary to translate a mask to a rgb tensor
idx2rgb = {key: value for key, value in enumerate(RGB_classes)}
rgb2idx = {v: k for k, v in idx2rgb.items()}

# Dict to map from label to rgb
rgb2label = dict(zip(Label_classes, RGB_classes))


################### Updated RGB Mappings after relabeling
# Get label RGB Mpaaings in all directions
# Define official RGB to Label mappings
RGB_classes = [
       (0, 0, 0), # ignore
       (255, 255, 225), # impervious
       (255,  0, 255), # building
       (255, 200, 0), # low vegetation
       (0,  0,  255), # water
       (0, 130, 0)] # trees
Label_classes = [
       "ignore",
       "impervious",
       "building",
       "low vegetation",
       "water",
       "trees"]

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
####################################  Dataset Class and DataLoader #############################
################################################################################################

def get_loveda_loaders(batch_size = 2):
    """
    Get data loaders for the LoveDA dataset. Loaders are initialized with suitable transforms functions.

    Args:
        batch_size (int, optional): The batch size for the data loaders. Defaults to 2.

    Returns:
        torch.utils.data.DataLoader: The data loader for the training set.
        torch.utils.data.DataLoader: The data loader for the test set.
    """
    # Define transforms
    def apply_train_transform(sample):
        # Define train transforms
        transform = A.Compose(
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

        # Get image and mask from LoveDA dataset entry
        mask = sample['mask']
        img = sample['image']
        # Convert to np.array
        mask = np.array(mask)

        ### Relabel mask to adjust to out use case
        mask[(mask == 1) | (mask == 3)] = 1 # Impervious surface = background + road
        #mask[mask == 2] = 2 # Building stays the same
        mask[(mask == 5) | (mask == 7)] = 3 # low vegetation = barren + agriculture
        mask[mask == 4] = 4 # Water stays the same
        mask[(mask == 6)] = 5 # forest

        img = np.array(img.permute(1,2,0))

        # transform image and mask
        transformed = transform(image = img, mask = mask)
        # return in correct format for the LoveDA dataset class of torchgeo
        return [transformed['image'], transformed['mask']]

    def apply_test_transform(sample):
        transform = A.Compose(
            [
            A.Resize(height=512, width=512),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
            ],
        )
        # Get image and mask from LoveDA dataset entry
        mask = sample['mask']
        img = sample['image']
        # Convert to np.array
        mask = np.array(mask)

        ### Relabel mask to adjust to out use case
        mask[(mask == 1) | (mask == 3)] = 1 # Impervious surface = background + road
        #mask[mask == 2] = 2 # Building stays the same
        mask[(mask == 5) | (mask == 7)] = 3 # low vegetation = barren + agriculture
        mask[mask == 4] = 4 # Water stays the same
        mask[(mask == 6)] = 5 # forest

        img = np.array(img.permute(1,2,0))

        # transform image and mask
        transformed = transform(image = img, mask = mask)
        # return in correct format for the LoveDA dataset class of torchgeo
        return [transformed['image'], transformed['mask']]

    # Download the dataset if not already downloaded
    # Instantiate Dataset Class
    train_dataset = LoveDA(
        root=path, 
        split='train', 
        scene=['urban', 'rural'], 
        transforms=lambda sample: apply_train_transform(sample), 
        download=True, 
        checksum=False)

    test_dataset = LoveDA(
        root=path, 
        split='val', 
        scene=['urban', 'rural'], 
        transforms=lambda sample: apply_test_transform(sample), 
        download=True, 
        checksum=False)

    print(f"""
        Train data: {len(train_dataset)}
        Validation data: {len(test_dataset)}
        """)
    
    # Instantiate Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)     

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