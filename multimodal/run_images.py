from data import *
from utils import *
from models import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir('../')
current_path = os.getcwd()
print('Current path: ', current_path)

pretrained_model = 'resnet50'
num_epochs = 2
verbose = True
learning_rate = 0.001

# if main == __ name __:
if __name__ == '__main__':
    
    # Reding the data
    print('Reading the data...')
    print
    info_jpg = pd.read_csv(current_path + '/dataset/mimic-cxr-2.0.0-metadata.csv')
    labels_data = pd.read_csv(current_path + '/dataset/mimic-cxr-2.0.0-chexpert.csv')
    image_files = list_images(current_path + '/dataset/files')
    image_labels_mapping = create_image_labels_mapping(image_files, labels_data, info_jpg)

    # Create dataframe  
    print('Creating dataframe...')
    df = pd.DataFrame.from_dict(image_labels_mapping, orient='index').reset_index()
    df['dicom_id'] = df['index'].apply(lambda x: x.split('/')[-1].split('.')[0])
    split = pd.read_csv(current_path + '/dataset/mimic-cxr-2.0.0-split.csv')
    df = pd.merge(df, split, on=['subject_id', 'study_id', 'dicom_id'], how = 'left')

    # Create training set
    print('Creating training set...')
    train_df = df[df['split'] == 'train']
    train_paths = train_df['index'].tolist()
    train_labels = train_df.iloc[:, 1:15].values.tolist()
    train_dict = create_image_labels_mapping(image_files, labels_data, info_jpg)

    # Create test set
    print('Creating test set...')
    test_df = df[df['split'] == 'test']
    test_paths = test_df['index'].tolist()
    test_labels = test_df.iloc[:, 1:15].values.tolist()
    test_dict = create_image_labels_mapping(image_files, labels_data, info_jpg)

    if pretrained_model == 'resnet50':
        print('Using ResNet50...')

        # Define your transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),                                               # Resize images to the size expected by ResNet
            transforms.ToTensor(),                                                       # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
        ])

        # Create datasets and dataloaders
        print('Creating datasets and dataloaders...')
        train_dataset = MedicalImagesDataset(train_dict, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
        test_dataset = MedicalImagesDataset(test_dict, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Iterate over batches of image,label pairs
        for i, (pa_images, lateral_images, labels) in enumerate(train_dataset):
            print('Posterior Anterior view: ', pa_images.shape)
            print('Lateral view', lateral_images.shape)
            break
        
        # Initiate loss, model, and optimizer
        class_counts = {8: 452, 0: 173, 1: 77, 7: 68, 9: 27, 6: 17, 3: 17, 
                        5: 16, 11: 14, 2: 14, 4: 9, 12: 7, 13: 3, 10: 2}
        class_weights = 1. / torch.tensor(list(class_counts.values()), dtype=torch.float)

        # Define model, loss, and optimizer
        print('Defining model, loss, and optimizer...')
        model = DualInputModel(model='resnet50', num_classes=14)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train and test model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = train_model(model, num_epochs, train_dataloader, criterion, optimizer, device, verbose = False)
        model = test_model(model, test_dataloader, device)
        torch.save(model.state_dict(), 'Resnet_model.pth')