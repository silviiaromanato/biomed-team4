'''
Grid search for radiology diagnosis using joint image-tabular encoders. 

You will neeed to install autoparse from GitHub using 
pip install git+https://github.com/latorrefabian/autoparse.git
'''

import os
import wandb
from autoparse import autoparse
import torch.optim as optim

from data import *
from train import *
from models import *

# Path to data and results directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Constants
NUM_LABELS = 3 # Neutral, Positive, Negative
NUM_CLASSES = 15 # Radiology diagnoses
TABULAR_DIM = 88 # Number of tabular features

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
                lr=0.001, 
                num_epochs=10,
                device='cpu',
                seed=0):
    '''
    Grid search for radiology diagnosis using joint image-tabular encoders. 
    '''

    # Create model
    tabular_params = {
        'dim_input': TABULAR_DIM,
        'hidden_dims': hidden_dims,
        'dropout_prob': dropout_prob,
        'batch_norm': batch_norm
    }
    model = JointEncoder(
        tabular=tabular, 
        tabular_params=tabular_params,
        vision=vision, 
        num_labels=NUM_LABELS,
        num_classes=NUM_CLASSES
    )
    model.to(device)

    # Freeze layers of vision encoder
    #if vision:
    #    for param in model.vision_encoder.parameters():
    #        param.requires_grad = False

    # Build W&B group
    build_group(tabular=tabular, vision=vision, tabular_params=tabular_params)

    # Load data (TODO)
    train_data, val_data, test_data = load_data(
        DATA_DIR, batch_size=64, num_workers=4, seed=seed, device=device)

    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model (TODO)
    train_results = train(model, criterion, optimizer, scheduler, train_data, val_data, num_epochs=num_epochs)
    wandb.log(train_results)

    # Evaluate model (TODO)
    eval_results = eval(model, criterion, test_data)
    wandb.log(eval_results)

    # Save model checkpoint
    save_path = os.path.join(CHECKPOINTS_DIR, 'final/' + wandb.run.name + '.pt')
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    kwargs = autoparse(grid_search, verbose=False)
    grid_search(**kwargs)
    