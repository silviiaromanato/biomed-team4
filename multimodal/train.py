'''
Grid search for radiology diagnosis using joint image-tabular encoders. 

You might need to install autoparse from GitHub using 
pip install git+https://github.com/latorrefabian/autoparse.git
'''

import os
import wandb
#from autoparse import autoparse
import torch.optim as optim
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
from transformers import TrainingArguments, Trainer

from train import *
from models import *
from data import *

# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

WANDB_PROJECT_NAME = 'BioMed'

# Path to data and results directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Constants
NUM_LABELS = 3 # Neutral, Positive, Negative
NUM_CLASSES = 14 # Radiology diagnoses
TABULAR_DIM = 88 # Number of tabular features

# ---------------------------------------- W&B FUNCTIONS ---------------------------------------- #

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

# ---------------------------------------- TRAINING FUNCTIONS ---------------------------------------- #


def compute_metrics(prediction): 
    ''' 
    Compares the diagnosis matrix to the ground truth. 
    Both prediction and labels are [batch_size, num_classes, num_labels] tensors.
    Computes accuracy, precision, recall, F1 score, AUC, and average precision.
    '''
    print('Computing metrics...')
    print('Prediction: ', prediction)
    labels = prediction.label_ids.flatten().detach().cpu().numpy()
    pred = prediction.predictions.flatten().detach().cpu().numpy()
    return {
        'accuracy': accuracy_score(labels, pred.round()),
        'precision': precision_score(labels, pred.round(), average='macro'),
        'recall': recall_score(labels, pred.round(), average='macro'),
        'f1': f1_score(labels, pred.round(), average='macro'),
        'auc': roc_auc_score(labels, pred),
        'ap': average_precision_score(labels, pred)
    }

class MultimodalTrainer(Trainer):
    '''
    Trainer class for joint image-tabular encoders. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        Computes loss for joint image-tabular encoders. 
        Diagnosis matrix computed by the model is compared to the ground truth.
        '''
        # Move inputs to GPU
        inputs = self._prepare_inputs(inputs)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

        # Predict diagnosis and compute loss
        labels = inputs.pop('label').float()
        diagnosis = model(**inputs)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(diagnosis, labels)
        return (loss, diagnosis) if return_outputs else loss


def train(model, train_data, val_data, test_data,
          output_dir, epochs=3, lr=2e-5, seed=0):
    '''
    Trains a Joint Encoder model. 
    W&B logging is enabled by default.

    Arguments: 
        model (JointEncoder): model to train
        train_data, val_data, test_data: Datasets
        run_name (str): Name of W&B run
        output_dir (str): Output directory for model checkpoints
        epochs (int): Number of training epochs
        lr (float): Learning rate
        seed (int): Random seed
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')

    training_args = TrainingArguments(

        # Training
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        dataloader_num_workers=0, # MIGHT NEED TO CHANGE THIS
        seed=seed,

        # Evaluation & checkpointing
        output_dir=output_dir,
        evaluation_strategy="epoch",
        eval_accumulation_steps=None,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # W&B logging
        report_to='wandb',
        logging_dir='./logs',
        logging_first_step=True,
        logging_steps=1,
        logging_strategy='epoch',
        run_name='test',
        use_mps_device=True # MIGHT NEED TO CHANGE THIS
    )
    # Train the model
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=train_data.collate_fn
    )
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=test_data)
    return eval_results

def grid_search(tabular=False, 
                vision=None, 
                hidden_dims=None, 
                dropout_prob=0.0, 
                batch_norm=True,
                lr=0.001, 
                num_epochs=10,
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

    # Freeze layers of vision encoder
    #if vision:
    #    for param in model.vision_encoder.parameters():
    #        param.requires_grad = False

    # Build W&B group
    build_group(tabular=tabular, vision=vision, tabular_params=tabular_params)

    # Load data
    tab_data, image_data = prepare_data()
    train_data, val_data, test_data = load_data(tab_data, image_data, vision=None)

    # Train model
    eval_results = train(model, train_data, val_data, test_data, CHECKPOINTS_DIR, epochs=num_epochs, lr=lr, seed=seed)
    return eval_results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--vision', type=str, default=None)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=None)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    #kwargs = autoparse(grid_search, verbose=False)
    grid_search(**vars(args))
    
    