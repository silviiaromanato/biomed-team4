'''
Grid search for radiology diagnosis using joint image-tabular encoders. 
'''

import os
import wandb
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
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
    logits = eval_preds.logits
    labels = eval_preds.label_ids
    preds = torch.argmax(torch.tensor(preds), dim=-1)
    labels = torch.argmax(torch.tensor(labels), dim=-1)
    metrics = {}
    for i, (disease, freqs) in enumerate(CLASS_FREQUENCIES.items()):
        inv_freqs = np.array([0 if x == 0 else 1/x for x in freqs.values()])
        inv_freqs = inv_freqs / inv_freqs.sum()
        metrics['acc_'+disease] = balanced_accuracy_score(labels[:, i], preds[:, i])
        metrics['macroF1_'+disease] = f1_score(labels[:, i], preds[:, i], average='macro')
        metrics['wF1_'+disease] = f1_score(labels[:, i], preds[:, i], average='weighted')
        metrics['wAUC_'+disease] = roc_auc_score(labels[:, i], logits[:, i, :], average='weighted', multi_class='ovr', labels=[0, 1, 2])

    metrics['acc_avg'] = np.mean([metrics['acc_'+disease] for disease in CLASS_FREQUENCIES.keys()])
    metrics['macroF1_avg'] = np.mean([metrics['macroF1_'+disease] for disease in CLASS_FREQUENCIES.keys()])
    metrics['wF1_avg'] = np.mean([metrics['wF1_'+disease] for disease in CLASS_FREQUENCIES.keys()])
    metrics['wAUC_avg'] = np.mean([metrics['wAUC_'+disease] for disease in CLASS_FREQUENCIES.keys()])

    return metrics


def create_trainer(model, train_data, val_data,
          run_name, output_dir, epochs=10, lr=1e-5,
          batch_size=16, weight_decay=1e-2, seed=0):
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, 
                                                num_training_steps=len(train_data)*epochs)

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
    trainer = create_trainer(model, train_data, val_data, run_name, CHECKPOINTS_DIR, 
                             epochs=num_epochs, lr=lr, weight_decay=weight_decay, seed=seed)
    print('Training:\tStarting training')
    trainer.train(ignore_keys_for_eval=['logits'])

    # Evaluate model
    if eval:
        print('Evaluation:\tEvaluating model on test set')
        eval_results = trainer.evaluate(eval_dataset=test_data)
        wandb.log(eval_results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabular', type=int, default=None)
    parser.add_argument('--vision', type=str, default=None)
    parser.add_argument('--hidden_dims', type=str, default=[256, 512])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.hidden_dims and type(args.hidden_dims) == str:
        args.hidden_dims = [int(x) for x in args.hidden_dims.split('-')]

    print(f'Cuda is available: {torch.cuda.is_available()}')

    grid_search(**vars(args))
    
    