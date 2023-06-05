# %%
# Custom utils
from potsdam_utils import *

# FCN Imports
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, FCN_ResNet50_Weights, FCN_ResNet101_Weights
# DeepLabV3 Imports
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.fcn import FCNHead
# Other torch imports
import torch.optim as optim
import torch.nn as nn

import seaborn as sns
import pandas as pd

import os
import shutil
import logging


# %%
# Load config file and get potsdam data path
import yaml
config_path = '/home/tu/tu_tu/tu_zxmav84/DS_Project/modules/config.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
path = config['data']['potsdam']

# ### Model setup

# %%
def calculate_accuracy(model, test_loader):

    # TODO: Implement Ccalculatation of accuracy per class

    print("\nEvaluating Accuracy on Test Set...")
    correct_pixels = 0
    num_pixels = 0

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            logits = model(data)['out']
            prob = nn.Softmax(logits).dim
            pred = torch.argmax(prob, dim = 1)
            correct_pixels += (pred.cpu() == label).sum()
            num_pixels += pred.numel()
        print(f"\n\tAccuracy: {np.round((correct_pixels/num_pixels)*100,2)}")
        logging.info(f"\n\tAccuracy: {np.round((correct_pixels/num_pixels)*100,2)}")

# %%
def train_epoch(model, train_loader, epoch):
    model.train()
    loss_sum = 0
    for batch_id, (data, label) in enumerate(train_loader):
        # Send data and label to DEVICE
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        # Forward Pass:
        output = model(data)['out']

        # Caluclate Loss
        loss = LOSS_FUNC(output, label.long())

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Comupte loss sum and count batches
        loss_sum += loss.item()
        batch_id += 1

        if batch_id%5==0:
            progress = f'Epoch: {epoch} | Batch {batch_id} / {len(train_loader)} | Loss: {np.round(loss_sum/(batch_id),4)}'
            print(progress)
            logging.info(progress)
            

# %%
def assign_device():
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

# %%
def train_model(DATASET = 'potsdam', MODEL_TYPE = 'FCN', BACKBONE = 'r101', NUM_EPOCHS=1, LEARNING_RATE = 0.01):
    #################### Catch NotImplemented ####################
    if DATASET not in ['potsdam']:
        print(f'Function not implemented for DATSET {DATASET}. Check spelling or datset selection')
        raise NotImplementedError
    if MODEL_TYPE not in ['FCN', 'DeepLabV3']:
        print(f'Function not implemented for MODEL_TYPE {MODEL_TYPE}. Check spelling or model selection')
        raise NotImplementedError
    if BACKBONE not in ['r50', 'r101']:
        print(f'Function not implemented for BACKBONE {BACKBONE}. Check spelling or backbone selection')
        raise NotImplementedError


    #################### Logging ####################
    base_dir = config['data']['model_weights']+ '/own_training/'
    specs_name = '_'.join([DATASET, MODEL_TYPE, BACKBONE, 'epochs'+str(NUM_EPOCHS), 'lr'+str(LEARNING_RATE).replace('.', '-')])
    save_dir = base_dir + specs_name

    # Create folder for this spec
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    os.mkdir(save_dir)

    # Configure logging
    # TODO Delete existing log file
    log_file = save_dir + '/' + specs_name+'.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Log Basis Training Specs
    logging.info(f'DATASET: {DATASET}')
    logging.info(f'MODEL_TYPE: {MODEL_TYPE}')
    logging.info(f'BACKBONE: {BACKBONE}')
    logging.info(f'NUM_EPOCHS: {NUM_EPOCHS}')
    logging.info(f'LEARNING_RATE: {LEARNING_RATE}')

    #################### DataLoaders ####################
    if DATASET == 'potsdam':
        train_loader, test_loader = get_potsdam_loaders(batch_size=2)

    # TODO: Implement DataLoader for LoveDA Dataset
    if DATASET == 'loveda':
        raise NotImplementedError



    #################### Instantiate Model ####################
    # Hyperparameters
    NUM_CLASSES = 6
    AUX_LAYER = True
    FROZEN = False

    #FCN 
    if MODEL_TYPE == 'FCN':
        if BACKBONE == 'r50':
            weights = FCN_ResNet50_Weights.DEFAULT # Default uses pretrained weights on COCO
            model = fcn_resnet50(weights=weights)
        elif BACKBONE == 'r101':
            weights = FCN_ResNet101_Weights.DEFAULT # Default uses pretrained weights on COCO
            model = fcn_resnet101(weights=weights)

        # Change output dimension of the net to different dimentsion to facilitate number of classes
        model.classifier[4] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1))

        if FROZEN: # freeze all weights except last layer, i.e. clasification head
            for layer_name, param in model.named_parameters():
                if 'classifier' not in layer_name:
                    param.requires_grad = False
        # Remove aux classifier if specified
        if not AUX_LAYER:
            model.aux_classifier = None
    #DeepLabV3   
    elif MODEL_TYPE == 'DeepLabV3':
        if BACKBONE == 'r50':
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        elif BACKBONE == 'r101':
            model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        model.classifier = DeepLabHead(2048, NUM_CLASSES)
        model.aux_classifier = FCNHead(1024, NUM_CLASSES)

    #################### Training Setup #################### 
    # Send model to DEVICE
    model = model.to(DEVICE)    

    # Optimization setup
    global optimizer
    global LOSS_FUNC
    optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    LOSS_FUNC = nn.CrossEntropyLoss()

    #################### Training #################### 
    calculate_accuracy(model, test_loader)
    print("Start Training ...")
    logging.info("Start Training ...")
    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_loader, epoch)
        calculate_accuracy(model, test_loader)

        #################### Saving ####################
        # Save model to disk
        torch.save(model, save_dir + '/' + specs_name+'_epoch'+str(epoch)+'.pth.tar')
        #torch.save(optimizer, save_dir + '/' + specs_name+'_optimizer.pth.tar')


