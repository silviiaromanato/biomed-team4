from data import *

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0, batch_norm=False, activation='relu'):
        super(FullyConnectedLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax()}
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
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_probs, batch_norm=True, activation='relu'):
        '''
        Fully-connected network with multiple layers.
        input_dim: input dimension
        output_dim: output dimension
        hidden_dims: list of hidden layer dimensions
        dropout_probs: list of dropout probabilities
        batch_norms: if True, use batch normalization
        activation: activation function
        '''
        super(FullyConnectedNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_probs = dropout_probs
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.layers = nn.ModuleList()
        self.layers.append(FullyConnectedLayer(input_dim, hidden_dims[0], dropout_probs[0], batch_norm, activation))
        for i in range(1, len(hidden_dims)):
            self.layers.append(FullyConnectedLayer(hidden_dims[i-1], hidden_dims[i], dropout_probs[i], batch_norm, activation))
        self.layers.append(FullyConnectedLayer(hidden_dims[-1], output_dim, dropout_probs[-1], batch_norm, 'sigmoid'))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x