from train_utils import *
import wandb

# Login in to wandb api
wandb.login()

# Specifying the search strategy
sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'Accuracy',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric


parameters_dict = {
    'BACKBONE':{'values':['r50','r101']},
    'LEARNING_RATE':{
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.5},
    'BATCH_SIZE':{'values':[2,4,6,8,10]}
    }

sweep_config['parameters'] = parameters_dict

def train_sweep(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        # Train multiple models on Potsdam
        train_model(DATASET='potsdam', MODEL_TYPE='DeepLabV3', BACKBONE=congig.BACKBONE, NUM_EPOCHS=10, LEARNING_RATE=config.LEARNING_RATE, BATCH_SIZE = config.BATCH_SIZE)

         wandb.log()

sweep_id = wandb.sweep(sweep_config, project="DS_Project")

wandb.agent(sweep_id, train_sweep, count=1)