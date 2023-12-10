'''
Grid search for radiology diagnosis using joint image-tabular encoders. 
'''

import os
import wandb
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score
from transformers import TrainingArguments, Trainer

from train import *
from models import *
from data import *

# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

CLASS_FREQUENCIES = [
    45808, 44845, 10778, 27018, 7179, 4390, 6284,
    51525, 75455, 54300, 2011, 16556, 10358, 66558
]

#os.environ['WANDB_SILENT'] = 'true'

# Path to data and results directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Constants
NUM_LABELS = 3 # Neutral, Positive, Negative
NUM_CLASSES = 14 # Radiology diagnoses

# ---------------------------------------- W&B FUNCTIONS ---------------------------------------- #

def build_group(tabular=False, 
                vision=None, 
                tabular_params=None,
                lr=0.001,
                weight_decay=0.01,
                ): 
    '''
    W&B group configuration for grid search.
    Initializes a new run with the specified group name and hyperparameters.

    Arguments: 
        - tabular (bool): Whether to use tabular data
        - vision (str): Type of vision encoder (Default: None --> No vision encoder)
        - tabular_params (dict): Parameters for tabular encoder {dim_input, hidden_dims, dropout_prob, batch_norm}
    '''
    if tabular == 0:
        tabular = False
    elif tabular == 1:
        tabular = True
    if tabular is None and vision is None: 
        raise ValueError('Error in build_group: tabular and/or vision must be specified.')  

    run_name = 'Tabular-' if tabular else ''
    run_name += f'{vision}-' if vision else ''
    if tabular:
        dims = '-'.join([str(x) for x in tabular_params["hidden_dims"]])
        run_name += f'{dims}'
        run_name += f'-p{tabular_params["dropout_prob"]}'
    if lr > 0:
        run_name += f'-lr{lr}'
    if weight_decay > 0:
        run_name += f'-wd{weight_decay}'
    
    config = {
        'tabular': tabular, 
        'vision': vision, 
        'tabular_params': tabular_params,
        'lr': lr,
        'weight_decay': weight_decay,
        }
    # Log to organization project
    print(f'W&B initialization: run {run_name}')
    wandb.init(group=run_name, 
               config=config, 
               project='biomed-team4', 
               entity='silvy-romanato'
               )
    return run_name

# ---------------------------------------- TRAINING FUNCTIONS ---------------------------------------- #

class MultimodalTrainer(Trainer):
    '''
    Custom trainer for multimodal models.
    Overriding for class-balanced cross-entropy loss.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor([1/x for x in CLASS_FREQUENCIES]).to(self.args.device)

    def compute_loss_balanced(self, model, inputs, return_outputs=False):
        '''
        Computes class-balanced cross-entropy loss.
        '''
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        loss = 0.0
        for i in range(NUM_CLASSES):
            loss += F.cross_entropy(logits[:, i, :], labels[:, i], weight=self.class_weights[i])
        return (loss, outputs) if return_outputs else loss
    

def compute_metrics(eval_preds): 
    ''' 
    Compares the diagnosis matrix to the ground truth. 
    Both prediction and labels are [batch_size, num_classes, num_labels] tensors.
    Computes accuracy, precision, recall, F1 score, AUC, and average precision.
    '''
    print('Computing metrics')
    preds = eval_preds.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = eval_preds.label_ids
    preds = torch.argmax(torch.tensor(preds), dim=-1)
    labels = torch.argmax(torch.tensor(labels), dim=-1)
    metrics = {}
    for i, disease in enumerate(CLASSES):
        metrics['acc_'+disease] = balanced_accuracy_score(labels[:, i], preds[:, i])
        metrics['f1_'+disease] = f1_score(labels[:, i], preds[:, i], average='weighted')

    # Compute average metrics
    accuracies = [metrics['acc_'+disease] for disease in CLASSES]
    f1_scores = [metrics['f1_'+disease] for disease in CLASSES]
    metrics['acc_avg'] = np.mean(accuracies)
    metrics['f1_avg'] = np.mean(f1_scores)
    print('Metrics:', metrics)
    return accuracies


def train(model, train_data, val_data, test_data,
          run_name, output_dir, epochs=10, lr=1e-5, weight_decay=1e-2, seed=0):
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
    print(f'Moving model to device: {device}')

    print('Training:\tInitializing training arguments')
    training_args = TrainingArguments(

        # Training
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        #warmup_steps=500,
        dataloader_num_workers=0, # MIGHT NEED TO CHANGE THIS
        seed=seed,

        # Evaluation & checkpointing
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        # W&B logging
        report_to='wandb',
        logging_dir='./logs',
        logging_first_step=True,
        logging_steps=1,
        logging_strategy='epoch',
        run_name=run_name,
        #use_mps_device=True # MIGHT NEED TO CHANGE THIS
    )

    print('Training:\tInitializing trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=train_data.collate_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print('Training:\tStarting training')
    trainer.train(
        ignore_keys_for_eval=['logits']
    )

    print('Evaluation:\tEvaluating model on test set')
    eval_results = trainer.evaluate(eval_dataset=test_data)
    return eval_results

def grid_search(tabular=False, 
                vision=None, 
                hidden_dims=None, 
                dropout_prob=0.0, 
                batch_norm=True,
                lr=0.001, 
                weight_decay=0.01,
                num_epochs=10,
                seed=0,
                **kwargs):
    '''
    Grid search for radiology diagnosis using joint image-tabular encoders. 
    '''
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    tabular_params = {
        'dim_input': NUM_FEATURES,
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
    run_name = build_group(tabular=tabular, vision=vision, tabular_params=tabular_params, 
                            lr=lr, weight_decay=weight_decay)

    # Load data
    tab_data, image_data = prepare_data()
    train_data, val_data, test_data = load_data(tab_data, image_data, vision=vision)

    # Train model
    eval_results = train(model, train_data, val_data, test_data, run_name, CHECKPOINTS_DIR, epochs=num_epochs, lr=lr, seed=seed)
    return eval_results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # store as boolean
    parser.add_argument('--tabular', type=int, default=None)
    parser.add_argument('--vision', type=str, default=None)
    parser.add_argument('--hidden_dims', type=str, default=[256, 512])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.hidden_dims and type(args.hidden_dims) == str:
        args.hidden_dims = [int(x) for x in args.hidden_dims.split('-')]

    print(f'Cuda is available: {torch.cuda.is_available()}')

    #kwargs = autoparse(grid_search, verbose=False)
    grid_search(**vars(args))
    
    