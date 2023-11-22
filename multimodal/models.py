from data import *

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