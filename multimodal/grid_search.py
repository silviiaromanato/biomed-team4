'''
Grid search for radiology diagnosis using joint image-tabular encoders. 

You will neeed to install autoparse from GitHub using 
pip install git+https://github.com/latorrefabian/autoparse.git
'''

import wandb
from autoparse import autoparse
import os
from data import *
from train import *

# Path to data and results directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')
GRID_PATH = os.path.join(GRID_DIR, 'grid.csv')

def build_group(tabular=False, 
                vision=None, 
                tabular_params=None): 
    '''
    W&B group configuration for grid search.
    Initializes a new run with the specified group name and hyperparameters.

    Arguments: 
        - tabular (bool): Whether to use tabular data
        - vision (str): Type of vision encoder (Default: None --> No vision encoder)
        - tabular_params (dict): Parameters for tabular encoder {dim_input, hidden_dims, dropout_prob, batch_norm}
    '''
    if tabular is None and vision is None: 
        raise ValueError('Error in build_group: tabular and/or vision must be specified.')  

    group = 'Tabular-' if tabular else ''
    group += f'{vision}' if vision else ''
    if tabular:
        group += f'_in{tabular_params["dim_input"]}'
        group += f'_hid{str(tabular_params["hidden_dims"])}'
        group += f'_p{tabular_params["dropout_prob"]}'
    
    config = {'tabular': tabular, 
              'vision': vision, 
              'tabular_params': tabular_params}
    wandb.init(group=group, config=config)

def grid_search(tabular=False, 
                vision=None, 
                hidden_dims=None, 
                dropout_prob=0.0, 
                batch_norm=True, 
                num_epochs=10):
    '''
    Grid search for radiology diagnosis using joint image-tabular encoders. 
    '''
    
    

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu')



if __name__ == '__main__':
    kwargs = autoparse(grid_search, verbose=False)
    grid_search(**kwargs)
    