# %%
# Basic Imports
import glob
import re
import os
import numpy as np
import random
from tqdm import tqdm
from glob import glob

# Torch Imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

# Image utils imports
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Custom imports that are shared with potsdam dataset
from potsdam_utils import denormalize_image


# %%
# Load config file and get orthophoto data path
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
root_dir = config['data']['orthophotos']

# %%
# Get path to all tiles stored in the "raw_tiles" folder
all_tiles = glob(root_dir + "/patched/*.tif")

# %%
# Custom Munich Dataset class
class MunichDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_filepath = self.file_paths[idx]

        image = np.array(Image.open(image_filepath))
    
        if self.transform is not None:
            transformed = self.transform(image=image)
            transformed_image = transformed['image']
            return transformed_image
        else:
            return image

# %%
# Test transform analogously to Potsdam test transform
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

# %%
def plot_example(img:np.ndarray, title:str=''):
    """
    Plot an example image with its denormalized version side by side.

    Args:
        img (numpy.ndarray): The input image to be plotted.
        title (str, optional): The title of the figure. Defaults to an empty string.

    Returns:
        None
    """
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(ToPILImage()(img))
    ax[1].imshow(ToPILImage()(denormalize_image(img)))
    [axs.axis('off') for axs in fig.get_axes()]

    # Set figure title
    plt.suptitle(title)
    fig.tight_layout()
    plt.show()

# %%
def get_munich_test_loader(batch_size):
    """
    Get the data loader for the Munich test dataset.

    Args:
        batch_size (int): The batch size for the data loader.

    Returns:
        torch.utils.data.DataLoader: The data loader for the Munich test dataset.
    """
    test_loader = DataLoader(MunichDataset(all_tiles, transform=test_transform),
                          batch_size = batch_size, 
                          num_workers = 2)
    return test_loader


