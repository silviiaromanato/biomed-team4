'''
Grid search for radiology diagnosis using joint image-tabular encoders. 
'''

import os
import wandb
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from models import *
from data import *

# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

CLASS_FREQUENCIES = {
    'Atelectasis': {0.0: 171692, 2.0: 45808, 1.0: 10327},
    'Cardiomegaly': {0.0: 176939, 2.0: 44845, 1.0: 6043},
    'Consolidation': {0.0: 212718, 2.0: 10778, 1.0: 4331},
    'Edema': {0.0: 187635, 2.0: 27018, 1.0: 13174},
    'Enlarged Cardiomediastinum': {0.0: 211273, 1.0: 9375, 2.0: 7179},
    'Fracture': {0.0: 222882, 2.0: 4390, 1.0: 555},
    'Lung Lesion': {0.0: 220402, 2.0: 6284, 1.0: 1141},
    'Lung Opacity': {0.0: 172471, 2.0: 51525, 1.0: 3831},
    'No Finding': {0.0: 152372, 2.0: 75455, 1.0: 0},
    'Pleural Effusion': {0.0: 167713, 2.0: 54300, 1.0: 5814},
    'Pleural Other': {0.0: 225051, 2.0: 2011, 1.0: 765},
    'Pneumonia': {0.0: 192980, 1.0: 18291, 2.0: 16556},
    'Pneumothorax': {0.0: 216335, 2.0: 10358, 1.0: 1134},
    'Support Devices': {0.0: 161032, 2.0: 66558, 1.0: 237}
    }

FREQUENCIES = [0.84384047, 0.1326398 , 0.02351973]
CLASS_WEIGHTS = torch.tensor([1.185058118257697, 6.403707815765645, 6.403707815765645])

CLASSES = list(CLASS_FREQUENCIES.keys())

#os.environ['WANDB_SILENT'] = 'true'

# Path to data and results directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')

NUM_FEATURES = 87   # Tabular features
NUM_LABELS = 3      # Neutral, Positive, Negative
NUM_CLASSES = 14    # Radiology diagnoses

WANDB_ENTITY = 'antoinebonnet'
WANDB_PROJECT = 'multimodal'

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
    tabular = bool(tabular)
    if tabular is False and vision is None: 
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
    
    config = {'tabular': tabular, 'vision': vision, 
              'tabular_params': tabular_params, 'lr': lr, 'weight_decay': weight_decay,}
    print(f'W&B initialization: run {run_name}')
    wandb.init(group=run_name, config=config, project=WANDB_PROJECT, entity=WANDB_ENTITY,)
    return run_name

# ---------------------------------------- TRAINING FUNCTIONS ---------------------------------------- #

class MultimodalTrainer(Trainer):
    '''
    Custom trainer for multimodal models.
    Overriding for class-balanced cross-entropy loss.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = CLASS_WEIGHTS

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        Computes class-balanced cross-entropy loss.
        '''
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels'].to(logits.device)
        loss = F.cross_entropy(
            logits.permute(0, 2, 1),
            labels.permute(0, 2, 1),
            weight=self.class_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss

def weighted_F1(preds, labels): 
    '''
    Given a tensor of predictions and a tensor of labels (1D and of same size)
    returns the weighted average F1 score by inverse label frequency. 
    '''
    weights = np.array([])
    
    for i in range(NUM_LABELS):
        freq = np.sum(np.where(labels == i, 1, 0))
        inv_freq = 0 if freq == 0 else 1/freq
        weights = np.append(weights, inv_freq)
    weights = weights / weights.sum()
    f1s = f1_score(labels, preds, average=None)

    # check if length f1s is equal to weights
    if len(f1s) != len(weights):
        # at the position where weights is 0, insert a 0 in f1s
        for i in range(len(weights)):
            if weights[i] == 0:
                f1s = np.insert(f1s, i, 0)
    weighted_f1 = np.sum(f1s * weights)
    return weighted_f1

def compute_metrics(eval_preds): 
    ''' 
    Compares the diagnosis matrix to the ground truth. 
    Both prediction and labels are [batch_size, num_classes, num_labels] tensors.

    Computes the following metrics (averaged over all diseases):
    - Balanced accuracy (average of recall for each class)
    - Macro F1 score (average of F1 scores for each class, same weight for each class)
    - Weighted F1 score (weighted average of F1 scores for each class, weighted by inverse class frequency)
    '''
    preds = eval_preds.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = eval_preds.label_ids

    # Convert one-hot encoding to class labels
    preds = torch.argmax(torch.tensor(preds), dim=-1) 
    labels = torch.argmax(torch.tensor(labels), dim=-1)

    # For each disease, compute metrics
    metrics = {}
    for i, disease in enumerate(CLASSES):
        metrics['acc_'+disease] = balanced_accuracy_score(labels[:, i], preds[:, i])        # Average of recall for each class
        metrics['macroF1_'+disease] = f1_score(labels[:, i], preds[:, i], average='macro')  # Same weight for each class
        metrics['wF1_'+disease] = weighted_F1(preds[:, i], labels[:, i])                    # Weighted average of F1 scores for each class
        metrics['macroPrec_'+disease] = precision_score(labels[:, i], preds[:, i], average='macro') # Same weight for each class
        metrics['macroRec_'+disease] = recall_score(labels[:, i], preds[:, i], average='macro') # Same weight for each class

    # Average over all diseases
    for metric in ['acc', 'macroF1', 'wF1', 'macroPrec', 'macroRec']:
        metrics[metric+'_avg'] = np.mean([metrics[metric+'_'+disease] for disease in CLASS_FREQUENCIES.keys()])
    return metrics


def create_trainer(model, train_data, val_data,
          run_name, output_dir, epochs=10, lr=1e-5,
          batch_size=16, weight_decay=1e-2, seed=0):
    '''
    Trains a Joint Encoder model. 
    W&B logging is enabled by default.

    Arguments:
        - model (JointEncoder): Joint encoder model
        - train_data (torch.utils.data.Dataset): Training data
        - val_data (torch.utils.data.Dataset): Validation data
        - run_name (str): W&B group name
        - output_dir (str): Path to directory where checkpoints will be saved
        - epochs (int): Number of epochs
        - lr (float): Learning rate
        - batch_size (int): Batch size
        - weight_decay (float): Weight decay
        - seed (int): Random seed
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Moving model to device: {device}')

    print('Training:\tInitializing optimizer and scheduler')
    params = [{'params': model.classifier.parameters(), 
                'lr': 0.001, 'weight_decay': 0.01}] # Fine-tuned on tabular only
    if model.tabular:
        params.append({
            'params': model.tabular_encoder.parameters(),
            'lr': 0.001, 'weight_decay': 0.01       # Fine-tuned on tabular only
            })
    if model.vision:
        params.append({'params': model.vision_encoder.parameters()}) 
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=len(train_data)*epochs)

    print('Training:\tInitializing training arguments')
    training_args = TrainingArguments(

        # Training
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0, # MIGHT NEED TO CHANGE THIS
        seed=seed,

        # Evaluation & checkpointing
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to='wandb',
        logging_dir='./logs',
        logging_first_step=True,
        logging_steps=1,
        logging_strategy='epoch',
        run_name=run_name,
        #use_mps_device=True # MIGHT NEED TO CHANGE THIS
    )

    print('Training:\tInitializing MultimodalTrainer')
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=train_data.collate_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        optimizers=(optimizer, scheduler)
    )
    return trainer

def grid_search(tabular=False, 
                vision=None, 
                hidden_dims=None, 
                dropout_prob=0.0, 
                batch_norm=True,
                lr=0.001, 
                weight_decay=0.01,
                num_epochs=10,
                seed=0,
                eval=False
                ):
    '''
    Grid search for radiology diagnosis using joint image-tabular encoders. 
    '''
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    tabular_params = {
        'dim_input': NUM_FEATURES,
        'hidden_dims': hidden_dims,
        'dropout_prob': dropout_prob,
        'batch_norm': batch_norm
    }

    print('Training:\tInitializing model')
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
    print('W&B:\tBuilding group')
    run_name = build_group(tabular=tabular, vision=vision, tabular_params=tabular_params, 
                            lr=lr, weight_decay=weight_decay)

    # Load data
    print('Data:\tLoading data')
    tab_data, image_data = prepare_data()
    train_data, val_data, test_data = load_data(tab_data, image_data, vision=vision)

    # Train model
    trainer = create_trainer(model, train_data, val_data, run_name, CHECKPOINTS_DIR, 
                             epochs=num_epochs, lr=lr, batch_size = 8, 
                             weight_decay=weight_decay, seed=seed)
    print('Training:\tStarting training')
    trainer.train(ignore_keys_for_eval=['logits'])

    # Evaluate model
    if eval:
        print('Evaluation:\tEvaluating model on test set')
        eval_results = trainer.evaluate(eval_dataset=test_data)
        wandb.log(eval_results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print('Grid search for radiology diagnosis using joint image-tabular encoders.')
    parser.add_argument('--tabular', type=int, default=None)
    parser.add_argument('--vision', type=str, default=None)
    print('Tabular encoder parameters: dim_input, hidden_dims, dropout_prob, batch_norm')
    parser.add_argument('--hidden_dims', type=str, default=[256, 512])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    print('Evaluation: evaluate model on test set')
    args = parser.parse_args()

    print('Parsing arguments')
    if args.hidden_dims and type(args.hidden_dims) == str:
        args.hidden_dims = [int(x) for x in args.hidden_dims.split('-')]

    print(f'Cuda is available: {torch.cuda.is_available()}')

    grid_search(**vars(args))
    
    