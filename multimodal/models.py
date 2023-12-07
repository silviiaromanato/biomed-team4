'''
Models for Multi-Modal Diagnosis from Radiology Images and Tabular metadata.
Vision encoders: ResNet50, DenseNet121, ViT
Tabular encoder: Fully-connected network
Joint encoder: Vision + Tabular encoders
'''

from data import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score


NUM_FEATURES = 87           # Number of tabular features
IMAGE_EMBEDDING_DIM = 512   # All vision encoders produce 512-dimensional embeddings

class FullyConnectedLayer(nn.Module):
    '''
    Single fully-connected Layer
    with batch normalization, dropout and ReLu activation.
    '''
    def __init__(self, 
                 input_dim,
                 output_dim, 
                 dropout_prob=0.0, 
                 batch_norm=True):
        super(FullyConnectedLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.activation = nn.ReLU()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU() 
        
    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.dropout_prob > 0.0:
            x = self.dropout(x)
        return x
    
class FullyConnectedNetwork(nn.Module):
    '''
    Fully-connected classifier network with multiple layers
    and batch normalization, ReLu, Dropout.   

    Args:
        dim_input (int): Input dimension
        hidden_dims (list): List of hidden layer dimensions
        dropout_prob (float): Dropout probability
        batch_norm (bool): Whether to use batch normalization 
    '''
    def __init__(self, 
                 dim_input,
                 hidden_dims,
                 dropout_prob=0.0, 
                 batch_norm=True):
        super(FullyConnectedNetwork, self).__init__()

        self.dim_input = dim_input
        self.hidden_dims = hidden_dims
        self.dims = [dim_input] + hidden_dims
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.sigmoid = nn.Sigmoid()

        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-1):
            self.layers.append(FullyConnectedLayer(self.dims[i], self.dims[i+1], dropout_prob, batch_norm))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ClassifierHead(nn.Module):
    '''
    Classifier head for multi-class multi-label prediction from embedding.
    Single layer -- with batch normalization, dropout and sigmoid activation.

    Args:
        dim_input (int): Input dimension
        num_classes (int): Number of classes
        num_labels (int): Number of labels for each class
    '''
    def __init__(self, dim_input, num_labels=3, num_classes=14):
        super(ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.dim_input = dim_input
        self.dim_output = num_classes * num_labels
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(self.dim_input, self.dim_output)

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, self.num_classes, self.num_labels)
        x = self.sigmoid(x)
        return x

    
class DualVisionEncoder(nn.Module):
    '''
    Dual vision encoders with dual input (PA and lateral images).
    Uses one vision encoder for each image, then concatenates the features.

    Args:
        vision (str): Type of vision encoder (resnet50, densenet121 or vit)
    '''
    def __init__(self, vision : str):
        super().__init__()

        # Load two pre-trained visual encoders
        if vision == 'resnet50':
            self.model_pa = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_lateral = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.num_features = self.model_pa.fc.in_features # 2048
            ''' DenseNet produces 1024-dimensional feature vectors. '''
        elif vision == 'densenet121':
            self.model_pa = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_lateral = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.num_features = self.model_pa.classifier.in_features # 1024
        elif vision == 'vit': 
            self.model_pa = ViTForImageClassification.from_pretrained(
                'google/vit-large-patch32-384', image_size=2500, patch_size=32, ignore_mismatched_sizes=True)
            self.model_lateral = ViTForImageClassification.from_pretrained(
                'google/vit-large-patch32-384', image_size=2500, patch_size=32, ignore_mismatched_sizes=True)
            self.num_features = self.model_pa.classifier.in_features # 768
        else: 
            raise ValueError(f'Vision encoder type {vision} not supported.')

        # Remove last classification layer (1000 classes in ImageNet)
        self.model_pa = nn.Sequential(*list(self.model_pa.children())[:-1])
        self.model_lateral = nn.Sequential(*list(self.model_lateral.children())[:-1])

        # Project to 512-dimensional embedding
        self.model_pa.add_module('embedding', nn.Linear(self.num_features, 512))
        self.model_lateral.add_module('embedding', nn.Linear(self.num_features, 512))

    def forward(self, x_pa, x_lat):
        features_pa = self.features_pa(x_pa)
        features_lateral = self.features_lateral(x_lat)
        combined_features = torch.cat((features_pa, features_lateral), dim=1)
        return combined_features
    
class JointEncoder(nn.Module):
    '''
    Joint Encoder: Encodes image and tabular data separately, 
    concatenates embeddings, and passes through fully connected classifier network.

    Args:
        tabular (bool): Whether to use tabular data
        tabular_params (dict): Parameters for tabular encoder {dim_input, hidden_dims, dropout_prob, batch_norm}
        vision (str): Type of vision encoder 'densenet121', 'resnet50' or 'vit'. Default: None --> No vision encoder
        num_labels (int): Number of labels for each class
        num_classes (int): Number of classes
    '''
    def __init__(self, 
                 tabular = True,
                 tabular_params = None,
                 vision = None,
                 num_labels=3,
                 num_classes=14
                 ):
        super(JointEncoder, self).__init__()

        self.tabular = tabular
        self.vision = vision
        self.dim_input = 0
        if not tabular and not vision: 
            raise ValueError('Must specify tabular and/or vision encoder.')
        
        if vision and vision not in ['resnet50', 'densenet121', 'vit']:
            raise ValueError(f'Vision encoder type {vision} not supported.')
        if vision:
            self.vision_encoder = DualVisionEncoder(vision)
            self.dim_input += IMAGE_EMBEDDING_DIM * 2

        if tabular:
            self.tabular_encoder = FullyConnectedNetwork(**tabular_params)
            self.dim_input += IMAGE_EMBEDDING_DIM

        self.classifier = ClassifierHead(self.dim_input, 
                                         num_labels=num_labels, 
                                         num_classes=num_classes)

    def forward(self, x_pa=None, x_lat=None, x_tab=None, **kwargs):
        '''
        Args:
            x_pa (tensor): PA image
            x_lat (tensor): Lateral image
            x_tab (tensor): Tabular features
        '''
        print('Forward:')
        print('\tx_pa: ', None if x_pa is None else x_pa.shape)
        print('\tx_lat: ', None if x_lat is None else x_lat.shape)
        print('\tx_tab: ', None if x_tab is None else x_tab.shape)
        for key, value in kwargs.items():
            print(f'\t{key}: ', None if value is None else value.shape)
            
        # Generate embeddings (image and/or tabular)
        if self.vision:
            if x_pa is None or x_lat is None:
                raise ValueError('Vision encoder is specified but no images are provided.')
            vision_embedding = self.vision_encoder(x_pa, x_lat)
        if self.tabular:
            if x_tab is None:
                raise ValueError('Tabular encoder is specified but no tabular data is provided.')
            tabular_embedding = self.tabular_encoder(x_tab)

        # Concatenate embeddings
        if self.vision and self.tabular:
            embedding = torch.cat((vision_embedding, tabular_embedding), dim=1)
        elif self.vision:
            embedding = vision_embedding
        elif self.tabular:
            embedding = tabular_embedding
        
        # Classify embeddings
        output = self.classifier(embedding)

        outputs = {}
        outputs['logits'] = output
        return output
