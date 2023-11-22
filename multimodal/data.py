import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class MimicDataset(torch.utils.data.Dataset):
    '''
    MIMIC Dataset. 
    Enables image and/or tabular data.
    '''
    def __init__(self, data_path, image=True, tabular=True):
        self.data_path = data_path
        self.image = image
        self.tabular = tabular
        
        self.data = pd.read_csv(data_path)
        self.labels = self.data['target'].values
        self.data.drop(['target'], axis=1, inplace=True)
        
        if self.image:
            self.image_data = self.data.iloc[:, :10000].values
            self.image_data = self.image_data.reshape(-1, 100, 100)
            self.image_data = torch.from_numpy(self.image_data).float()
            self.image_data = self.image_data.unsqueeze(1)
        if self.tabular:
            self.tabular_data = self.data.iloc[:, 10000:].values
            self.tabular_data = torch.from_numpy(self.tabular_data).float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.image and self.tabular:
            return self.image_data[idx], self.tabular_data[idx], self.labels[idx]
        elif self.image:
            return self.image_data[idx], self.labels[idx]
        elif self.tabular:
            return self.tabular_data[idx], self.labels[idx]
        
def train_val_test_split(dataset, val_size=0.2, test_size=0.2, shuffle=True):
    '''
    Split dataset into train, validation, and test sets.
    '''
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    val_split = int(np.floor(val_size * dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    train_indices, val_indices, test_indices = indices[val_split+test_split:], indices[:val_split], indices[val_split:test_split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=test_sampler)
    return train_loader, val_loader, test_loader