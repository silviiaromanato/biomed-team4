'''
Data loading and preprocessing for MIMIC-CXR and MIMIC-IV. 
Multimodal data loader and dataset classes. 
Train/val/test splitting. 
'''

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor
import pickle
from tqdm import tqdm

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

NUM_LABELS = 3                          # Number of labels for each class
NUM_CLASSES = 14                        # Number of classes
IMAGE_SIZE = 384                        # All images are resized to 384 x 384
NORM_MEAN = [0.4734, 0.4734, 0.4734]    # MIMIC-CXR mean (based on 2GB of images)
NORM_STD = [0.3006, 0.3006, 0.3006]     # MIMIC-CXR std (based on 2GB of images)


# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

DATA_PATH = 'data/'
TABULAR_PATH = os.path.join(DATA_PATH, 'mimic-iv')
IMAGES_PATH = os.path.join(DATA_PATH, 'mimic-cxr')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed_data')

METADATA_PATH = os.path.join(IMAGES_PATH, 'mimic-cxr-2.0.0-metadata.csv.gz')
LABELS_PATH = os.path.join(IMAGES_PATH, 'mimic-cxr-2.0.0-chexpert.csv.gz')

TAB_PATH = os.path.join(PROCESSED_PATH, 'tab_data_total.csv')
TAB_TRAIN_PATH = os.path.join(PROCESSED_PATH, 'tab_data_train.csv')
TAB_VAL_PATH = os.path.join(PROCESSED_PATH, 'tab_data_val.csv')
TAB_TEST_PATH = os.path.join(PROCESSED_PATH, 'tab_data_test.csv')

LABELS_TRAIN_PATH = os.path.join(PROCESSED_PATH, 'labels_train.csv')
LABELS_VAL_PATH = os.path.join(PROCESSED_PATH, 'labels_val.csv')
LABELS_TEST_PATH = os.path.join(PROCESSED_PATH, 'labels_test.csv')

# ---------------------------------------- HELPER FUNCTIONS ---------------------------------------- #

def list_images(base_path):
    '''
    Recursively lists all image files starting from the base path.
    Assumes that images have extensions typical for image files (e.g., .jpg, .jpeg, .png).
    '''
    image_files = []
    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

def load_tabular_data():
    '''
    Load tabular data: admissions, patients, services, image metadata
    '''
    admissions_path = os.path.join(TABULAR_PATH, 'admissions.csv.gz')
    patients_path = os.path.join(TABULAR_PATH, 'patients.csv.gz')
    services_path = os.path.join(TABULAR_PATH, 'services.csv.gz')
    metadata_path = os.path.join(TABULAR_PATH, 'mimic-cxr-2.0.0-metadata.csv.gz')

    if not os.path.exists(admissions_path):
        raise ValueError(f'Admissions file not found in {admissions_path}.')
    if not os.path.exists(patients_path):
        raise ValueError(f'Patients file not found in {patients_path}.')
    if not os.path.exists(services_path):
        raise ValueError(f'Services file not found in {services_path}.')
    if not os.path.exists(metadata_path):
        raise ValueError(f'Image metadata file not found in {metadata_path}.')
    
    admissions = pd.read_csv(admissions_path)
    patients = pd.read_csv(patients_path)
    services = pd.read_csv(services_path)
    metadata = pd.read_csv(metadata_path)
    return admissions, patients, services, metadata

def load_images_data():
    '''
    Load image data: labels, image files, image metadata
    '''
    if not os.path.exists(LABELS_PATH):
        raise ValueError(f'Labels file not found in {LABELS_PATH}.')
    if not os.path.exists(IMAGES_PATH):
        raise ValueError(f'Images folder not found in {IMAGES_PATH}.')
    if not os.path.exists(METADATA_PATH):
        raise ValueError(f'Image metadata file not found in {METADATA_PATH}.')
    
    labels_data = pd.read_csv(LABELS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    image_files = list_images(IMAGES_PATH)
    if image_files == []:
        raise ValueError(f'No image files found in {IMAGES_PATH}.')
    
    return labels_data, image_files, metadata


def create_image_labels_mapping(image_files, labels_data, info_data):
    '''
    Create a mapping from image files to their corresponding labels and view positions.

    Parameters:
    - image_files: List of image file paths
    - labels_data: DataFrame containing label information
    - info_data: DataFrame containing additional information like ViewPosition

    Returns:
    A dictionary with image file paths as keys and dicts with labels and ViewPosition as values.
    '''
    image_labels_mapping = {}

    for image_path in tqdm(image_files):
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
        if len(labels_row) > 1:
            raise ValueError(f'More than one label row found for {image_path}.')
        
        if not labels_row.empty and not view_info.empty:
            labels = labels_row.iloc[0].to_dict()
            labels['ViewPosition'] = view_info['ViewPosition'].values[0]
            image_labels_mapping[image_path] = labels

    return image_labels_mapping
        

def join_multimodal(labels_data, image_files, info_jpg, tab_data):
    '''
    Join the tabular data with the image data.
    Returns: 
        tab_data (DataFrame): tabular data
        dict_img (dict): keys = image file paths and values = dicts with labels and ViewPosition
    '''
    print('Join multimodal data')
    # Tabular data
    print('Tabular data')
    tab_data.loc[:, 'study_id'] = tab_data['study_id'].astype(int).astype(str)
    tab_data.loc[:, 'subject_id'] = tab_data['subject_id'].astype(int).astype(str)

    # Image data
    print('Image data')
    image_labels_mapping = create_image_labels_mapping(image_files, labels_data, info_jpg)
    df_img = pd.DataFrame.from_dict(image_labels_mapping, orient='index').reset_index()
    df_img['study_id'] = df_img['study_id'].astype(int).astype(str)
    df_img['subject_id'] = df_img['subject_id'].astype(int).astype(str)

    # Keep only PA and LATERAL images
    print('Keep only PA and LATERAL images')
    df_img = df_img[df_img['ViewPosition'].isin(['PA', 'LATERAL'])]

    # Group by study_id and subject_id and ViewPosition and keep the first row
    print('Group by study_id and subject_id and ViewPosition and keep the first row')
    df_img = df_img.groupby(['study_id', 'subject_id', 'ViewPosition']).first().reset_index()

    # Function to check if both PA and Lateral images are present
    def has_both_views(group):
        return 'PA' in group['ViewPosition'].values and 'LATERAL' in group['ViewPosition'].values

    # Filter the DataFrame
    print('Filter the DataFrame')
    df_img = df_img.groupby(['study_id', 'subject_id']).filter(has_both_views)

    # Filter on study
    print('Filter on study')
    common_data = set(tab_data['study_id']).intersection(set(df_img['study_id']))
    tab_data = tab_data[tab_data['study_id'].isin(common_data)]
    df_img = df_img[df_img['study_id'].isin(common_data)]

    # Filter on subject
    print('Filter on subject')
    common_data = set(tab_data['subject_id']).intersection(set(df_img['subject_id']))
    tab_data = tab_data[tab_data['subject_id'].isin(common_data)]
    df_img = df_img[df_img['subject_id'].isin(common_data)]
    print(f'Number of samples:\tTabular: {len(tab_data)}\tImage: {len(df_img)}')

    # Return the image data to a dictionary
    print('Return the image data to a dictionary')
    dict_img = df_img.set_index('index').T.to_dict()

    return tab_data, dict_img
    
# ---------------------------------------- PREPROCESSING ---------------------------------------- #

def preprocess_tabular():
    if os.path.exists(TAB_PATH):
        print(f'Loading:\tPre-processed tabular data from {TAB_PATH}.')
        return pd.read_csv(TAB_PATH)
    print(f'Loading:\tPreprocessing tabular data, saving to {TAB_PATH}.')

    # PREPROCESS ADMISSIONS AND PATIENTS
    # Drop columns, merge admissions and patients, convert dates to datetime
    admissions, patients, services, metadata = load_tabular_data()
    admissions_cleaned = admissions.drop(columns=[
        'dischtime', 'deathtime', 'discharge_location', 'edregtime', 'edouttime', 'hospital_expire_flag'])
    patients_cleaned = patients.drop(columns=['dod'])
    tabular_data = pd.merge(admissions_cleaned, patients_cleaned, on='subject_id', how='left')
    tabular_data['admittime'] = pd.to_datetime(tabular_data['admittime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # PREPROCESS IMAGE METADATA
    # Keep only images with both PA and LATERAL views
    len_old = len(metadata)
    groups = metadata.groupby('study_id')
    study_ids = []
    for study_id, group in groups: 
        if len(group) > 1:
            if 'PA' in group['ViewPosition'].values and 'LATERAL' in group['ViewPosition'].values:
                study_ids.append(study_id)
    metadata = metadata[metadata['study_id'].isin(study_ids)]
    metadata = metadata[metadata['ViewPosition'].isin(['PA', 'LATERAL'])]
    metadata = metadata.groupby(['study_id', 'ViewPosition']).first().reset_index()
    len_new = len(metadata)
    print(f'Number of studies with Lateral and PA image: {len_new} ({len_new/len_old*100:.2f}%)')

    # FILTER IMAGES
    # Keep only radiology studies with known date
    metadata_time = pd.DataFrame(columns=['subject_id', 'StudyDate', 'StudyTime', 'dicom_id', 'study_id', 'ViewPosition'])
    metadata_time['dicom_id'] = metadata['dicom_id']
    metadata_time['study_id'] = metadata['study_id']
    metadata_time['ViewPosition'] = metadata['ViewPosition']
    metadata_time['subject_id'] = metadata['subject_id']
    metadata_time['StudyDate'] = pd.to_datetime(metadata['StudyDate'], format='%Y%m%d').dt.date
    metadata_time['StudyTime'] = metadata['StudyTime'].astype(str).str.split('.').str[0]
    metadata_time['StudyTime'] = pd.to_datetime(metadata['StudyTime'], format='%H%M%S', errors='coerce').dt.time
    old_len = len(metadata_time)
    metadata_time = metadata_time.dropna(subset=['StudyDate', 'StudyTime'])
    new_len = len(metadata_time)
    print(f'Dropped {old_len - new_len} rows because the date or time of the study was not known')
    metadata_time['StudyDateTime'] = pd.to_datetime(
        metadata_time['StudyDate'].astype(str) + ' ' + metadata_time['StudyTime'].astype(str))
    metadata_time = metadata_time.drop(columns=['StudyDate', 'StudyTime'])

    # MERGE IMAGE METADATA WITH ADMISSION/PATIENTS DATA
    # Keep only images after admittime, keep only the latest admittime
    merged = pd.merge(metadata_time, tabular_data, on='subject_id', how='left')
    merged = merged[merged['StudyDateTime'] >= merged['admittime']]
    merged = merged.sort_values('admittime').groupby('dicom_id').last().reset_index()

    # MERGE TABULAR METADATA & SERVICES
    # Keep only services before StudyDateTime, keep only the latest transfer
    unique_services = services['curr_service'].unique()
    services['transfertime'] = pd.to_datetime(services['transfertime'], format='%Y-%m-%d %H:%M:%S').dt.date
    tms = pd.merge(merged, services, on=['hadm_id', 'subject_id'], how='left')
    tms = tms[tms['transfertime'] <= tms['StudyDateTime']]
    tms = tms.sort_values('transfertime').groupby('dicom_id').last().reset_index()
    for service in unique_services:
        tms[service] = 0
    for index, row in tms.iterrows():
        tms.loc[index, row['curr_service']] = 1
    tms = tms.drop(columns=['curr_service', 'transfertime', 'prev_service'])
    print('Number of pictures (front and lateral together): ', len(merged))
    print('Number of unique subjects: ', len(merged['subject_id'].unique()))
    print('Number of unique studies: ', len(merged['study_id'].unique()))
    tms = tms[tms['ViewPosition']=='PA']
    tabular = tms.drop(
        columns=['dicom_id', 'ViewPosition', 'StudyDateTime', 
                 'hadm_id', 'admit_provider_id', 'anchor_year', 'admittime'])

    # ONE HOT ENCODING OF CATEGORICAL VARIABLES
    tabular = pd.get_dummies(
        tabular, 
        columns=['admission_type', 'admission_location', 'language', 
                 'marital_status', 'race', 'gender', 
                 'anchor_year_group', 'anchor_year_group', 'insurance'])

    # SAVE PREPROCESSED DATA
    tabular.to_csv(TAB_PATH, index=False)
    return tabular
    
def preprocess_labels(): 
    '''
    Preprocess labels:
    - Positive: 1 -> 2
    - Uncertain: -1 -> 1
    - Missing == Negative: NaN -> 0
    '''
    if not os.path.exists(LABELS_PATH):
        raise ValueError(f'Labels file not found in {LABELS_PATH}.')
    labels = pd.read_csv(LABELS_PATH)
    labels = labels.replace(1, 2)
    labels = labels.replace(-1, 1) 
    labels = labels.fillna(0)      
    return labels


def split(tabular, labels, val_size=0.1, test_size=0.15, seed=42):
    '''
    Split tabular data and labels into train, val, and test sets.
    '''
    paths = [TAB_TRAIN_PATH, TAB_VAL_PATH, TAB_TEST_PATH, 
             LABELS_TRAIN_PATH, LABELS_VAL_PATH, LABELS_TEST_PATH]
    
    if all([os.path.exists(path) for path in paths]):
        print('Splitting:\tLoading pre-processed train, val, and test sets.')
        tabular_train = pd.read_csv(TAB_TRAIN_PATH)
        tabular_val = pd.read_csv(TAB_VAL_PATH)
        tabular_test = pd.read_csv(TAB_TEST_PATH)
        labels_train = pd.read_csv(LABELS_TRAIN_PATH)
        labels_val = pd.read_csv(LABELS_VAL_PATH)
        labels_test = pd.read_csv(LABELS_TEST_PATH)

    else:
        print('Splitting:\tTabular data and labels into train, val, and test sets.')
        # Split the study_ids into train, val, and test sets
        study_ids = tabular['study_id'].unique()
        np.random.seed(seed)
        np.random.shuffle(study_ids)
        num_study_ids = len(study_ids)
        num_val = int(num_study_ids * val_size)
        num_test = int(num_study_ids * test_size)
        study_ids_train = study_ids[num_val + num_test:]
        study_ids_val = study_ids[:num_val]
        study_ids_test = study_ids[num_val:num_val + num_test]

        # Get the tabular data and labels for the train, val, and test sets
        tabular_train = tabular[tabular['study_id'].isin(study_ids_train)]
        tabular_val = tabular[tabular['study_id'].isin(study_ids_val)]
        tabular_test = tabular[tabular['study_id'].isin(study_ids_test)]
        labels_train = labels[labels['study_id'].isin(study_ids_train)]
        labels_val = labels[labels['study_id'].isin(study_ids_val)]
        labels_test = labels[labels['study_id'].isin(study_ids_test)]

        # Save the train, val, and test sets
        tabular_train.to_csv(TAB_TRAIN_PATH, index=False)
        tabular_val.to_csv(TAB_VAL_PATH, index=False)
        tabular_test.to_csv(TAB_TEST_PATH, index=False)
        labels_train.to_csv(LABELS_TRAIN_PATH, index=False)
        labels_val.to_csv(LABELS_VAL_PATH, index=False)
        labels_test.to_csv(LABELS_TEST_PATH, index=False)

        # Check proportions of total, train, val, and test sets
        total_len = len(tabular_train) + len(tabular_val) + len(tabular_test)
        print('Total set: ', total_len)
        print('Percent train: ', len(tabular_train) / total_len)
        print('Percent val: ', len(tabular_val) / total_len)
        print('Percent test: ', len(tabular_test) / total_len)

    return tabular_train, tabular_val, tabular_test, labels_train, labels_val, labels_test


# ---------------------------------------- DATA LOADING ---------------------------------------- #

def transform_image(image_size, vision=None, augment=True): 
    '''
    Defines the image transformation pipeline. 
    1. Augmentation (flips, rotations) (only for training)
    2. Crop a square from the (non-square) image
    3. Resize to IMAGE_SIZE x IMAGE_SIZE
    4. Convert to tensor
    5. Normalize (with ImageNet mean and std)
    '''
    transforms = []
    size = min(image_size) # Get minimum of image height and width to crop to square

    # Augmentation (flips, rotations)
    if augment:
        transforms.append(RandomRotation(10))
        transforms.append(RandomVerticalFlip())
        transforms.append(RandomHorizontalFlip())

    transforms.append(CenterCrop((size, size)))

    if vision == 'vit':
        processor = ViTImageProcessor.from_pretrained(
            'google/vit-large-patch32-384', 
            do_normalize=False, 
            image_mean=NORM_MEAN, 
            image_std=NORM_STD, 
            return_tensors='pt')
        transforms.append(lambda x: processor(x).pixel_values[0])
    else: 
        transforms.append(Resize((IMAGE_SIZE, IMAGE_SIZE)))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=NORM_MEAN, std=NORM_STD))
    return Compose(transforms)

class MultimodalDataset(Dataset):
    '''
    Dataset class for MIMIC-CXR and MIMIC-IV.
    Handles both tabular data and images.
    '''
    def __init__(self, vision, data_dict, tabular, augment=True):
        self.vision = vision
        self.data_dict = data_dict
        self.tabular = tabular

        if vision is not None: 
            self.transform = lambda img_size: transform_image(img_size, vision=vision, augment=augment)
        
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                        'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        
        # Organize paths by subject_id and study_id
        self.organized_paths = self._organize_paths()
        #print(f'Number of samples: {len(self.organized_paths)}')

        # Filter out pairs where both images are None
        self.organized_paths = {k: v for k, v in self.organized_paths.items() if v['PA'] is not None and v['LATERAL'] is not None}

    def _organize_paths(self):
        organized = {}
        for path in self.data_dict.keys():
            parts = path.split(os.sep)
            subject_id = parts[-3][1:]
            study_id = parts[-2][1:]
            key = (subject_id, study_id)
            if key not in organized:
                organized[key] = {'PA': None, 'LATERAL': None}
            view_position = self.data_dict[path]['ViewPosition']
            if view_position in ['PA', 'LATERAL']:
                organized[key][view_position] = path

        return organized

    def __len__(self):
        return len(self.organized_paths)

    def _load_and_process_image(self, path):
        if path:
            image = Image.open(path).convert('RGB')
        else:
            image = Image.new('RGB', (self.size, self.size))
        image = self.transform(image.size)(image)
        return image

    def __getitem__(self, idx):

        if idx >= len(self.organized_paths):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.organized_paths)} samples.")
        
        # Get the subject_id and study_id for this index
        subject_study_pair = list(self.organized_paths.keys())[idx]

        # Get tabular data
        subject_id, study_id = subject_study_pair
        tabular_row = self.tabular[(self.tabular['subject_id'] == subject_id) & 
                                (self.tabular['study_id'] == study_id)]
        tabular_row = tabular_row.drop(['subject_id', 'study_id'], axis=1).values
        tabular_row = np.where(tabular_row == False, 0.0, tabular_row)
        tabular_row = np.where(tabular_row == True, 1.0, tabular_row)

        # Convert to tensor, if this cannot be done throw an error and print the row
        tabular_tensor = torch.from_numpy(tabular_row.astype(float))

        # Get the paths for the PA and Lateral images
        pa_path = self.organized_paths[subject_study_pair]['PA']
        lateral_path = self.organized_paths[subject_study_pair]['LATERAL']

        # Get labels from the image data
        labels_path = pa_path if pa_path else lateral_path
        if not labels_path:
            raise ValueError(f'No labels path found for {subject_study_pair}.')
        labels = self.data_dict[labels_path]
        label_values = [labels[class_name] if not np.isnan(labels[class_name]) else 0 for class_name in self.classes]
        label_values = torch.tensor(label_values, dtype=torch.float32).unsqueeze(0)
        if torch.any(label_values < 0):
            print(f'Negative label values for {subject_study_pair}: {label_values}')
        label_tensor = torch.nn.functional.one_hot(label_values.to(torch.int64), num_classes=NUM_LABELS).squeeze(0).float()
        
        inputs = {'x_tab': tabular_tensor, 'labels': label_tensor}
        
        # Load and process PA and Lateral images
        if self.vision is not None:
            pa_image = self._load_and_process_image(pa_path) \
                if pa_path else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            lateral_image = self._load_and_process_image(lateral_path) \
                if lateral_path else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            if not isinstance(pa_image, torch.Tensor):
                pa_image = torch.tensor(pa_image)
            if not isinstance(lateral_image, torch.Tensor):
                lateral_image = torch.tensor(lateral_image)
            inputs['x_pa'] = pa_image
            inputs['x_lat'] = lateral_image
        return inputs

    def collate_fn(self, batch):
        inputs = {}
        if 'x_tab' in batch[0]:
            inputs['x_tab'] = torch.stack([x['x_tab'] for x in batch if 'x_tab' in x])
        
        if 'labels' in batch[0]:
            inputs['labels'] = torch.stack([x['labels'] for x in batch if 'labels' in x])

        if self.vision is not None:
            if 'x_pa' in batch[0]:
                inputs['x_pa'] = torch.stack([x['x_pa'] for x in batch if 'x_pa' in x])

            if 'x_lat' in batch[0]:
                inputs['x_lat'] = torch.stack([x['x_lat'] for x in batch if 'x_lat' in x])
        return inputs

def prepare_data(): 
    '''
    Load and pre-process tabular data and labels.
    Split into train/val/test sets.
    Filter images based on tabular data.
    '''
    pickle_tab_path = os.path.join(PROCESSED_PATH, 'tab_data.pickle')
    pickle_img_path = os.path.join(PROCESSED_PATH, 'image_data.pickle')

    if os.path.exists(pickle_tab_path) and os.path.exists(pickle_img_path):
        with open(pickle_tab_path, 'rb') as handle:
            tab_data = pickle.load(handle)
        with open(pickle_img_path, 'rb') as handle:
            image_data = pickle.load(handle)
        return tab_data, image_data
    
    print(f'PREPARING DATA')
    # Load and pre-process tabular data and labels
    tabular = preprocess_tabular()
    labels = preprocess_labels()

    # Split tabular and labels into train/val/test sets
    tab_train, tab_val, tab_test, lab_train, lab_val, lab_test = split(tabular, labels)
    
    # Load image labels, files and metadata
    print('Loading:\tImage data (labels, files, metadata).')
    labels_data, image_files, metadata = load_images_data()

    print('Joining:\tIntersection of tabular and image data.')
    tab_data_train, image_data_train = join_multimodal(lab_train, image_files, metadata, tab_train)
    tab_data_val, image_data_val = join_multimodal(lab_val, image_files, metadata, tab_val)
    tab_data_test, image_data_test = join_multimodal(lab_test, image_files, metadata, tab_test)
    tab_data = {'train': tab_data_train, 'val': tab_data_val, 'test': tab_data_test}
    image_data = {'train': image_data_train, 'val': image_data_val, 'test': image_data_test}

    with open(pickle_tab_path, 'wb') as handle:
        pickle.dump(tab_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pickle_img_path, 'wb') as handle:
        pickle.dump(image_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return tab_data, image_data


def load_data(tab_data, image_data, vision=None):
    '''
    Create datasets for each split.
    Arguments: 
        tab_data (dict): Dictionary with keys = 'train', 'val', 'test' and values = tabular data
        image_data (dict): Dictionary with keys = 'train', 'val', 'test' and values = image data
        vision (str): Type of vision encoder 'resnet50', 'densenet121' or 'vit' (Default: None --> No images)
    '''
    print(f'LOADING DATA (vision: {vision})')
    print(f'Loaded image data:\tTrain: {len(image_data["train"])}\tValidation: {len(image_data["val"])}\tTest: {len(image_data["test"])} samples.')
    print(f'Loaded tabular data: \tTrain: {len(tab_data["train"])}\tValidation: {len(tab_data["val"])}\tTest: {len(tab_data["test"])} samples.')
    train_data = MultimodalDataset(vision, image_data['train'], tab_data['train'], augment=True)
    val_data = MultimodalDataset(vision, image_data['val'], tab_data['val'], augment=False)
    test_data = MultimodalDataset(vision, image_data['test'], tab_data['test'], augment=False)
    print(f'Created datasets:\tTrain: {len(train_data)}\tValidation: {len(val_data)}\tTest: {len(test_data)} samples.')
    return train_data, val_data, test_data


if __name__ == '__main__': 

    tab_data, image_data = prepare_data()
    tab_data_train, tab_data_val, tab_data_test = tab_data['train'], tab_data['val'], tab_data['test']
    image_data_train, image_data_val, image_data_test = image_data['train'], image_data['val'], image_data['test']

    # Print the shapes of the dataframes
    print(f'Tabular data\nTrain: {tab_data_train.shape}\nVal: {tab_data_val.shape}\nTest: {tab_data_test.shape}')
    print(f'Image data\nTrain: {len(image_data_train)}\nVal: {len(image_data_val)}\nTest: {len(image_data_test)}')

    # Save the dataframes
    tab_data_train.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_train.csv'), index=False)
    tab_data_val.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_val.csv'), index=False)
    tab_data_test.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_test.csv'), index=False)

    # Save the dictionaries
    np.save(os.path.join(PROCESSED_PATH, 'image_data_train.npy'), image_data_train)
    np.save(os.path.join(PROCESSED_PATH, 'image_data_val.npy'), image_data_val)
    np.save(os.path.join(PROCESSED_PATH, 'image_data_test.npy'), image_data_test)

    # Delete not matched images
    all_images = set(list(image_data_train.keys()) + list(image_data_val.keys()) + list(image_data_test.keys()))
    _, image_files, _ = load_images_data()
    for image_file in image_files:
        if image_file not in all_images:
            os.remove(image_file)


