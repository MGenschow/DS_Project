# %%
# Basic Imports
from glob import glob
import re
import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import check_random_state
from tqdm import tqdm

# Torch Imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
# Other torch imports
import torch
import torch.optim as optim
import torch.nn as nn

# Image utils imports
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# %%
# Load config file and get potsdam data path
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
orthophoto_dir = config['data']['orthophotos']

# %% [markdown]
# ### General Helper functions

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    check_random_state(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(123)

set_seed(1234)
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


# %% [markdown]
# ### Path Helper Functions

# %%
def get_file_paths():
    """
    Retrieves the file paths of image and mask files in the labeling_subset directory.

    Returns:
        dict: A dictionary containing the file paths of image and mask files.
              The dictionary is integer-indexed for easy sampling using an integer index.
              The keys are integers representing the index, and the values are tuples
              containing the image file path and the corresponding mask file path.

    """
    mask_files = glob(orthophoto_dir + '/labeling_subset/final_masks/*.tif')
    image_files = glob(orthophoto_dir + '/labeling_subset/images/*.tif')

    print(f"Indexing files in orthophotos/labeling_subset... \nFound: \t {len(image_files)} Images \n\t {len(mask_files)} Mask")

    # Get base name of all files and create dict with image and mask file paths
    pattern = '\d+_+\d+_patch_\d{1,2}_\d{1,2}'
    patch_base_names = [re.search(pattern, mask_files[i]).group(0) for i in range(len(mask_files))]
    # The dictionary is integer-indexed to allow the dataset __getitem__ class to sample using an integer idx
    path = orthophoto_dir + '/labeling_subset'
    file_paths = {i:(path+'/images/'+name+'.tif',path+'/final_masks/'+name+'.tif') for i, name in enumerate(patch_base_names)}
    return file_paths


# %%
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
    np.random.seed(42)
    train_keys, test_keys = train_test_split(list(file_paths.keys()), test_size=test_size)
    train_dict = {i:file_paths[key] for i,key in enumerate(train_keys)}
    test_dict = {i:file_paths[key] for i,key in enumerate(test_keys)}
    print(f"Length of all files: {len(file_paths)}")
    print(f"Length of train ({len(train_dict)}) and test ({len(test_dict)}): {len(train_dict)+len(test_dict)}")
    return train_dict, test_dict

# %% [markdown]
# ### Dataloader

# %%
class MunichTuningDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_filepath = self.file_paths[idx][0]
        mask_filepath = self.file_paths[idx][1]

        image = np.array(Image.open(image_filepath))
        label_mask = np.array(Image.open(mask_filepath))
        # Convert RGB mask to label mask
    
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label_mask)
            transformed_image = transformed['image']
            transformed_label_mask = transformed['mask']
            return transformed_image, transformed_label_mask
        else:
            return image, label_mask

# %%
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

# %%
def get_munich_tuning_loaders(batch_size=2):
    file_paths = get_file_paths()
    train_dict, test_dict = train_test_split(file_paths, test_size=0.2)
    train_loader = DataLoader(MunichTuningDataset(train_dict, transform=train_transform), 
                            batch_size = batch_size, 
                            num_workers = 2)
    test_loader = DataLoader(MunichTuningDataset(test_dict, transform=test_transform),
                            batch_size = batch_size, 
                            num_workers = 2)
    print(f"Length of train loader: {len(train_loader)}; Length of test loader: {len(test_loader)} with batch size {batch_size}")

    return train_loader, test_loader

# %% [markdown]
### Training Utils

# %%
def assign_device():
    """
    Assigns the appropriate device (CPU or GPU) for training based on the availability of CUDA-enabled devices.

    Returns:
        None
    """
    global DEVICE
    # Device setup
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    print(f"Using {DEVICE} as DEVICE")

assign_device()


def calculate_accuracy(model, test_loader, num_classes):
    """
    Calculate the pixel-wise accuracy of a model on the test set, 
    along with per-class accuracy, mean accuracy, and average accuracy.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        num_classes: The number of target classes.

    Returns:
        DataFrame containing per-class accuracies, mean accuracy, and average accuracy.
    """
    print("\nEvaluating Accuracy on Test Set...")

    correct_pixels = 0
    num_pixels = 0

    # Maintain a list of counters for each class
    correct_pixels_per_class = [0] * num_classes
    num_pixels_per_class = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(DEVICE)
            logits = model(data)['out']
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            correct_pixels += (pred.cpu() == label).sum().item()
            num_pixels += np.prod(label.shape)

            # Calculate per class accuracy
            for i in range(num_classes):
                if i in [0,4]:
                    continue
                else:
                    correct_pixels_per_class[i] += ((pred.cpu() == label) & (label == i)).sum().item()
                    num_pixels_per_class[i] += (label == i).sum().item()

    # Calculate and store per class accuracies
    accuracy_dict = {}
    per_class_acc = []
    for i in range(num_classes):
        if i in [0,4]:
                continue
        else:
            if num_pixels_per_class[i] > 0:
                accuracy = np.round((correct_pixels_per_class[i]/num_pixels_per_class[i])*100, 2)
                per_class_acc.append(accuracy)
                accuracy_dict[Label_classes[i]] = accuracy
            else:
                accuracy_dict[Label_classes[i]] = 'No instances in the test set'

    # Calculate and store mean accuracy
    if per_class_acc:
        mean_acc = np.mean(per_class_acc)
        accuracy_dict['Mean Accuracy'] = mean_acc

    # Calculate and store average accuracy
    avg_acc = correct_pixels / num_pixels * 100
    accuracy_dict['Average Accuracy'] = avg_acc

    # Convert dictionary to pandas DataFrame and return
    df = pd.DataFrame(list(accuracy_dict.items()), columns=['Metric', 'Accuracy (%)'])
    
    return df

def train_epoch(model, train_loader, epoch):
    """
    Trains the model for one epoch using the provided training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader object that provides the training data.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.train()
    loss_sum = 0
    for batch_id, (data, label) in enumerate(train_loader):
        # Send data and label to DEVICE
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.cuda.amp.autocast(): 
            # Forward Pass:
            output = model(data)['out']

            # Caluclate Loss
            loss = LOSS_FUNC(output, label.long())

        # Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Comupte loss sum and count batches
        loss_sum += loss.item()
        batch_id += 1

        if batch_id%5==0:
            progress = f'Epoch: {epoch} | Batch {batch_id} / {len(train_loader)} | Loss: {np.round(loss_sum/(batch_id),4)}'
            print(progress)


def train_model(model, train_loader, test_loader, LEARNING_RATE = 0.01, NUM_EPOCHS=1):
    #################### Training Setup #################### 
    # Send model to DEVICE
    model = model.to(DEVICE)    

    # Optimization setup
    global optimizer
    global LOSS_FUNC
    global scaler
    optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    #weights = [0, 1, 2, 2, 0, 2]
    #label_weights = torch.FloatTensor(weights).cuda()
    #LOSS_FUNC = nn.CrossEntropyLoss(weight = label_weights)

    LOSS_FUNC = nn.CrossEntropyLoss()
    # use torch grad scaler to speed up training and make it more stable
    scaler = torch.cuda.amp.GradScaler()

    #################### Training #################### 
    full_result = calculate_accuracy(model, test_loader, 6)
    display(full_result)
    print("Start Training ...")
    
    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_loader, epoch)
        result = calculate_accuracy(model, test_loader, 6)
        result.rename(columns = {'Accuracy (%)':f"Epoch {epoch}"}, inplace = True)
        full_result = pd.concat([full_result, result.iloc[:,1]], axis = 1)
        display(full_result)

        #################### Saving ####################
        # Save model to disk
        #torch.save(model, save_dir + '/' + specs_name+'_epoch'+str(epoch)+'.pth.tar')
        #torch.save(optimizer, save_dir + '/' + specs_name+'_optimizer.pth.tar'




# %%
#train_loader, test_loader = get_munich_tuning_loaders(batch_size=2)

# %% [markdown]
# ### Test DataLoader Output
# %%
# iterate over the train_loader
# for i, (images, masks) in enumerate(test_loader):
#     # stop after the first batch
#     if i > 3:
#         break

#     batch_size = images.shape[0]

#     for j in range(batch_size):
#         # select the j-th image and mask from the batch
#         image = ToPILImage()(images[j])
#         mask_img = ToPILImage()(masks[j])
#         mask = masks[j].numpy()

#         # PyTorch dataloaders usually return images in (C, H, W) format,
#         # so we need to transpose this to (H, W, C) for matplotlib to display it correctly

#         # Create an empty RGB mask
#         mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

#         # Map each label index to its RGB equivalent
#         for idx, rgb in idx2rgb.items():
#             mask_rgb[mask == idx] = rgb
        
#         # Plotting
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#         ax[0].imshow(image)  # display the image
#         ax[0].set_title(f'Image {j+1}')
#         ax[0].axis('off')

#         ax[1].imshow(mask_rgb)  # display the mask
#         ax[1].set_title(f'Label Mask {j+1}')
#         ax[1].axis('off')

#         plt.show()

# %%
