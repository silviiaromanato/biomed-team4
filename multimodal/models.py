from data import *
from torchvision.models import densenet121, DenseNet121_Weights, ResNet50_Weights, resnet50
from transformers import ViTForImageClassification
from torchvision import models
import torch.nn as nn
import torch


NUM_LABELS = 3 # Neutral, Positive, Negative
NUM_CLASSES = 15 # Radiology diagnoses
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
    def __init__(self, num_labels=3, num_classes=15):
        super(ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels
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
        vision_type (str): Type of vision encoder (resnet50, densenet121 or vit)
    '''
    def __init__(self, vision_type : str):
        super().__init__()

        # Load two pre-trained visual encoders
        if vision_type == 'resnet50':
            self.model_pa = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_lateral = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.num_features = self.model_pa.fc.in_features
        elif vision_type == 'densenet121':
            self.model_pa = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_lateral = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.num_features = self.model_pa.classifier.in_features
        elif vision_type == 'vit':
            self.model_pa = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.model_lateral = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.num_features = self.model_pa.classifier.in_features
        else: 
            raise ValueError(f'Vision encoder type {vision_type} not supported.')

        # Remove last classification layer (1000 classes in ImageNet)
        self.model_pa = nn.Sequential(*list(self.model_pa.children())[:-1])
        self.model_lateral = nn.Sequential(*list(self.model_lateral.children())[:-1])

    def forward(self, x_pa, x_lateral):
        features_pa = self.features_pa(x_pa)
        features_lateral = self.features_lateral(x_lateral)
        combined_features = torch.cat((features_pa, features_lateral), dim=1)
        return combined_features
    
class JointEncoder(nn.Module):
    '''
    Joint Encoder: Encodes image and tabular data separately, 
    concatenates embeddings, and passes through fully connected classifier network.

    Args:
        tabular (bool): Whether to use tabular data
        tabular_params (dict): Parameters for tabular encoder {dim_input, hidden_dims, dropout_prob, batch_norm}
        vision_type (str): Type of vision encoder (Default: None --> No vision encoder)
        num_labels (int): Number of labels for each class
        num_classes (int): Number of classes
    '''
    def __init__(self, 
                 tabular = True,
                 tabular_params = None,
                 vision_type = None,
                 num_labels=3, 
                 num_classes=15
                 ):
        super(JointEncoder, self).__init__()

        if not tabular and not vision_type: 
            raise ValueError('Must specify tabular and/or vision encoder.')
        
        if vision_type and vision_type not in ['resnet50', 'densenet121', 'vit']:
            raise ValueError(f'Vision encoder type {vision_type} not supported.')
        if vision_type:
            self.vision_encoder = DualVisionEncoder(vision_type)

        if tabular:
            self.tabular_encoder = FullyConnectedNetwork(**tabular_params)
        
        self.classifier = ClassifierHead(num_labels=num_labels, num_classes=num_classes)

    def forward(self, image_pa=None, image_lateral=None, tabular=None):
        '''
        Args:
            image_pa (tensor): PA image
            image_lateral (tensor): Lateral image
            tabular (tensor): Tabular features
        '''
        # Generate embeddings (image and/or tabular)
        if self.vision_encoder:
            if image_pa is None or image_lateral is None:
                raise ValueError('Must provide image input.')
            image_embedding = self.vision_encoder(image_pa, image_lateral)
        if self.tabular_encoder:
            if tabular is None:
                raise ValueError('Must provide tabular input.')
            tabular_embedding = self.tabular_encoder(tabular)

        # Concatenate embeddings
        if self.vision_encoder and self.tabular_encoder:
            embedding = torch.cat((image_embedding, tabular_embedding), dim=1)
        elif self.vision_encoder:
            embedding = image_embedding
        elif self.tabular_encoder:
            embedding = tabular_embedding
        
        # Classify embeddings
        output = self.classifier(embedding)
        return output