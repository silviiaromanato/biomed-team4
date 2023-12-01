import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import math
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import os

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

def list_images(base_path):
    """
    Recursively lists all image files starting from the base path.
    Assumes that images have extensions typical for image files (e.g., .jpg, .jpeg, .png).
    """
    image_files = []
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

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
        
class MedicalImagesDataset(Dataset):
    def __init__(self, data_dict, size=224, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.size = size
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                        'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        # Organize paths by subject_id and study_id
        self.organized_paths = self._organize_paths()
        # Filter out pairs where both images are None
        self.organized_paths = {k: v for k, v in self.organized_paths.items() if v['PA'] is not None or v['Lateral'] is not None}

    def _organize_paths(self):
        organized = {}
        for path in self.data_dict.keys():
            parts = path.split(os.sep)
            subject_id = parts[-3][1:]
            study_id = parts[-2][1:]
            key = (subject_id, study_id)

            if key not in organized:
                organized[key] = {'PA': None, 'Lateral': None}

            view_position = self.data_dict[path]['ViewPosition']
            if view_position in ['PA', 'Lateral']:
                organized[key][view_position] = path

        return organized

    def __len__(self):
        return len(self.organized_paths)

    def _load_and_process_image(self, path):
        if path:
            image = Image.open(path).convert('RGB')
        else:
            # Create a blank (black) image if path is None
            image = Image.new('RGB', (self.size, self.size))

        image = image.resize((self.size, self.size))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image

    def __getitem__(self, idx):
        subject_study_pair = list(self.organized_paths.keys())[idx]
        pa_path = self.organized_paths[subject_study_pair]['PA']
        lateral_path = self.organized_paths[subject_study_pair]['Lateral']

        # Load and process PA and Lateral images
        pa_image = self._load_and_process_image(pa_path)
        lateral_image = self._load_and_process_image(lateral_path)

        # Use one of the available paths to get labels (assuming they are the same for both views)
        labels_path = pa_path if pa_path else lateral_path

        if not labels_path:
            # Skip this patient if both PA and Lateral images are missing
            return None

        labels = self.data_dict[labels_path]
        label_values = [labels[class_name] if not np.isnan(labels[class_name]) else 0 for class_name in self.classes]
        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        return pa_image, lateral_image, label_tensor

    

def create_image_labels_mapping(image_files, labels_data, info_data):
    """
    Create a mapping from image files to their corresponding labels and view positions.

    Parameters:
    - image_files: List of image file paths
    - labels_data: DataFrame containing label information
    - info_data: DataFrame containing additional information like ViewPosition

    Returns:
    A dictionary with image file paths as keys and dicts with labels and ViewPosition as values.
    """
    image_labels_mapping = {}

    for image_path in image_files:
        # Extract subject_id, study_id, and dicom_id from the file path
        parts = image_path.split(os.sep)
        subject_id = parts[-3][1:]  # Assuming the subject_id is prefixed with a character to be removed
        study_id = parts[-2][1:]    # Assuming the study_id is prefixed with a character to be removed
        dicom_id = parts[-1][:-4]   # Assuming the file extension is 4 characters long (e.g., .jpg)

        # Find the corresponding row in the labels CSV
        labels_row = labels_data[
            (labels_data['subject_id'] == int(subject_id)) &
            (labels_data['study_id'] == int(study_id))
        ]

        # Find the view position info
        view_info = info_data[
            (info_data['subject_id'] == int(subject_id)) &
            (info_data['study_id'] == int(study_id)) &
            (info_data['dicom_id'] == dicom_id)
        ]

        # Assuming there is only one match, get the labels and view position
        if not labels_row.empty and not view_info.empty:
            labels = labels_row.iloc[0].to_dict()
            labels['ViewPosition'] = view_info['ViewPosition'].values[0]
            image_labels_mapping[image_path] = labels

    return image_labels_mapping
        
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