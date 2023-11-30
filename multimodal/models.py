from data import *
from torchvision.models import densenet121, DenseNet121_Weights, ResNet50_Weights, resnet50
from torchvision import models
import torch.nn as nn


class Classifier(nn.Module):
    '''
    Fully-connected classifier network.
    Converts an input embedding into a multi-class multi-label prediction.
    Batch normalization, ReLu, Dropout.   

    Args:
        embed_size (int): Input embedding size
        hidden_dims (list): List of hidden layer dimensions
        num_classes (int): Number of classes to predict
        num_labels (int): Number of labels to predict
        dropout_prob (float): Dropout probability
    '''
    def __init__(self, 
                 embed_size, 
                 hidden_dims,
                 num_classes=1, 
                 num_labels=1, 
                 dropout_prob=0.0):
        super(Classifier, self).__init__()

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(embed_size))
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.layers.append(nn.ReLU())
            if dropout_prob > 0.0:
                self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class DualInputModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        if model == 'resnet50':
            self.features_pa = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features_lateral = models.resnet50(weights=ResNet50_Weights.DEFAULT)

            # Replace the classifier in both ResNet models
            # The nn.Identity() layer removes the final classification layer of the ResNet models (classifying 1000 different classes in ImageNet).
            num_features = self.features_pa.fc.in_features
            self.features_pa.fc = nn.Identity()
            self.features_lateral.fc = nn.Identity()
            
        elif model == 'densenet121':
            # Load pre-trained DenseNet models
            self.features_pa = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.features_lateral = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
            # Replace the classifier in both DenseNet models
            # The nn.Identity() layer removes the final classification layer of the DenseNet models (classifying 1000 different classes in ImageNet).
            num_features = self.features_pa.classifier.in_features
            self.features_pa.classifier = nn.Identity()
            self.features_lateral.classifier = nn.Identity()
        
        # Combine features from both models for classification
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x_pa, x_lateral):
        # Forward pass for both images through their respective DenseNet models
        features_pa = self.features_pa(x_pa)
        features_lateral = self.features_lateral(x_lateral)
        
        # Concatenate the features
        combined_features = torch.cat((features_pa, features_lateral), dim=1)
        
        # Classify the combined features
        out = self.classifier(combined_features)
        return out

class FullyConnectedLayer(nn.Module):
    '''
    Fully Connected Layer with batch normalization, dropout and custom activation.
    '''
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 dropout_prob=0.0, 
                 batch_norm=False, 
                 activation='relu'):
        super(FullyConnectedLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        activations = {'relu': nn.ReLU(), 
                       'sigmoid': nn.Sigmoid(), 
                       'tanh': nn.Tanh(), 
                       'softmax': nn.Softmax()}
        if activation in activations:
            self.activation = activations[activation]
        else: 
            raise ValueError(f'Activation function {activation} not supported.')
        
    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        return x
    
class FullyConnectedNetwork(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims, 
                 dropout_prob, 
                 batch_norm=True, 
                 activation='relu'):
        '''
        Fully-connected network with multiple layers.
        '''
        super(FullyConnectedNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.layers = nn.ModuleList()
        self.layers.append(FullyConnectedLayer(input_dim, hidden_dims[0], 
                                               dropout_prob, batch_norm, activation))
        for i in range(1, len(hidden_dims)):
            self.layers.append(FullyConnectedLayer(hidden_dims[i-1], hidden_dims[i], 
                                                   dropout_prob, batch_norm, activation))
        self.layers.append(FullyConnectedLayer(hidden_dims[-1], output_dim, 
                                               dropout_prob, batch_norm, 'sigmoid'))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class JointEncoder(nn.Module):
    def __init__(self, 
                 image_encoder_type, 
                 image_input_dim, 
                 tabular_input_dim, 
                 output_dim, 
                 image_hidden_dims, 
                 tabular_hidden_dims, 
                 image_dropout_prob, 
                 tabular_dropout_prob, 
                 batch_norm=True, 
                 activation='relu'):
        '''
        Joint Encoder: Encodes image and tabular data separately, 
        concatenates embeddings, and passes through fully connected classifier network.
        '''
        super(JointEncoder, self).__init__()
        
        self.image_input_dim = image_input_dim
        self.image_hidden_dims = image_hidden_dims
        self.image_dropout_probs = image_dropout_prob

        self.tabular_input_dim = tabular_input_dim
        self.tabular_hidden_dims = tabular_hidden_dims
        self.tabular_dropout_probs = tabular_dropout_prob

        self.output_dim = output_dim

        self.batch_norm = batch_norm
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        # Image encoder
        if image_encoder_type == 'resnet':
            self.image_encoder = ResNet(image_input_dim, output_dim, image_hidden_dims, 
                                        image_dropout_prob, batch_norm, activation)
        elif image_encoder_type == 'densenet':
            self.image_encoder = DenseNet(image_input_dim, output_dim, image_hidden_dims, 
                                          image_dropout_prob, batch_norm, activation)
        elif image_encoder_type == 'vit':
            self.image_encoder = VisionTransformer(image_input_dim, output_dim, image_hidden_dims, 
                                                   image_dropout_prob, batch_norm, activation)
        else:
            raise ValueError(f'Image encoder type {image_encoder_type} not supported.')

        self.tabular_encoder = FullyConnectedNetwork(
            tabular_input_dim, output_dim, tabular_hidden_dims, 
            tabular_dropout_prob, batch_norm, activation)
        
        self.classifier = FullyConnectedNetwork(output_dim*2, output_dim, [output_dim], [0.0], batch_norm, activation)
        
    def forward(self, image, tabular):
        image_embedding = self.image_encoder(image)
        tabular_embedding = self.tabular_encoder(tabular)
        embedding = torch.cat((image_embedding, tabular_embedding), dim=1)
        output = self.classifier(embedding)
        output = self.sigmoid(output)
        return output