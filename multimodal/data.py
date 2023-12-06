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

# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

DATA_PATH = '../data/'
TABULAR_PATH = os.path.join(DATA_PATH, 'mimic-iv')
IMAGES_PATH = os.path.join(DATA_PATH, 'mimic-cxr')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed_data')

METADATA_PATH = os.path.join(IMAGES_PATH, 'mimic-cxr-2.0.0-metadata.csv')
LABELS_PATH = os.path.join(IMAGES_PATH, 'mimic-cxr-2.0.0-chexpert.csv')

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
    metadata_path = os.path.join(TABULAR_PATH, 'mimic-cxr-2.0.0-metadata.csv')

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
    image_files = list_images(os.path.join(IMAGES_PATH, 'files'))
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
        

def filter_images(labels_data, image_files, info_jpg, tab_data):
    '''
    Filter the images and labels based on the tabular data.
    Used to optimize storage.
    '''
    # Tabular data
    tab_data['study_id'] = tab_data['study_id'].astype(int).astype(str)
    tab_data['subject_id'] = tab_data['subject_id'].astype(int).astype(str)

    # Image data
    image_labels_mapping = create_image_labels_mapping(image_files, labels_data, info_jpg)
    df_img = pd.DataFrame.from_dict(image_labels_mapping, orient='index').reset_index()
    df_img['study_id'] = df_img['study_id'].astype(int).astype(str)
    df_img['subject_id'] = df_img['subject_id'].astype(int).astype(str)

    # Filter on study
    common_data = set(tab_data['study_id']).intersection(set(df_img['study_id']))
    tab_data = tab_data[tab_data['study_id'].isin(common_data)]
    df_img = df_img[df_img['study_id'].isin(common_data)]

    # Filter on subject
    common_data = set(tab_data['subject_id']).intersection(set(df_img['subject_id']))
    tab_data = tab_data[tab_data['subject_id'].isin(common_data)]
    df_img = df_img[df_img['subject_id'].isin(common_data)]
    print(f'Number of studies in tabular data: {len(tab_data)} and in image data: {len(df_img)}')

    # Return the imae data to a dictionary
    dict_img = df_img.set_index('index').T.to_dict()

    return tab_data, dict_img


# ---------------------------------------- PREPROCESSING ---------------------------------------- #

def preprocess_tabular():
    if os.path.exists(TAB_PATH):
        print(f'Loading pre-processed tabular data from {TAB_PATH}.')
        return pd.read_csv(TAB_PATH)
    print(f'Preprocessing tabular data, saving to {TAB_PATH}.')

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

    if not all([os.path.exists(path) for path in paths]):

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

    else:   
        tabular_train = pd.read_csv(TAB_TRAIN_PATH)
        tabular_val = pd.read_csv(TAB_VAL_PATH)
        tabular_test = pd.read_csv(TAB_TEST_PATH)
        labels_train = pd.read_csv(LABELS_TRAIN_PATH)
        labels_val = pd.read_csv(LABELS_VAL_PATH)
        labels_test = pd.read_csv(LABELS_TEST_PATH)

    return tabular_train, tabular_val, tabular_test, labels_train, labels_val, labels_test


# ---------------------------------------- DATA LOADING ---------------------------------------- #

class MultimodalDataset(Dataset):
    '''
    Dataset class for MIMIC-CXR and MIMIC-IV.
    Handles both tabular data and images.
    '''
    def __init__(self, vision, data_dict, tabular):
        self.vision = vision
        self.data_dict = data_dict
        self.tabular = tabular
        self.size = 224

        if vision == 'vit':
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.transform = lambda x: self.processor(x, return_tensors='pt').pixel_values.squeeze(0)

        elif vision in ['resnet50', 'dense121']:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),  # Resize to 224x224
                transforms.ToTensor(),                      # Convert images to PyTorch tensors
                transforms.Normalize(                       # Normalize with ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],    
                    std=[0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError(f'Vision encoder {vision} not supported.')
        
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                        'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        
        # Organize paths by subject_id and study_id
        self.organized_paths = self._organize_paths()
        #print(f'Number of samples: {len(self.organized_paths)}')

        # Filter out pairs where both images are None
        self.organized_paths = {k: v for k, v in self.organized_paths.items() \
                                if v['PA'] is not None and v['LATERAL'] is not None}

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
        if self.vision == 'vit':
            image = self.processor(image, return_tensors='pt').pixel_values.squeeze(0)
        image = self.transform(image)
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
        tabular_tensor = torch.tensor(tabular_row, dtype=torch.float32)

        if not self.vision:
            return tabular_tensor, None, None, None

        # Get the paths for the PA and Lateral images
        pa_path = self.organized_paths[subject_study_pair]['PA']
        lateral_path = self.organized_paths[subject_study_pair]['LATERAL']

        # Load and process PA and Lateral images
        pa_image = self._load_and_process_image(pa_path)
        lateral_image = self._load_and_process_image(lateral_path)

        # Use one of the available paths to get labels
        labels_path = pa_path if pa_path else lateral_path
        if not labels_path:
            return None # Skip if both are missing

        labels = self.data_dict[labels_path]
        label_values = [labels[class_name] if not np.isnan(labels[class_name]) else 0 for class_name in self.classes]
        label_tensor = torch.tensor(label_values, dtype=torch.float32)

        return pa_image, lateral_image, label_tensor, tabular_tensor


def prepare_data(): 
    '''
    Load and pre-process tabular data and labels.
    Split into train/val/test sets.
    Filter images based on tabular data.
    '''
    # Load and pre-process tabular data and labels
    tabular = preprocess_tabular()
    labels = preprocess_labels()

    # Split tabular and labels into train/val/test sets
    tab_train, tab_val, tab_test, lab_train, lab_val, lab_test = split(tabular, labels)
    
    # Load image labels, files and metadata
    labels_data, image_files, metadata = load_images_data()

    # Get intersection of tabular and image data
    tab_data_train, image_dict_train = filter_images(lab_train, image_files, metadata, tab_train)
    tab_data_val, image_dict_val = filter_images(lab_val, image_files, metadata, tab_val)
    tab_data_test, image_dict_test = filter_images(lab_test, image_files, metadata, tab_test)

    return tab_data_train, tab_data_val, tab_data_test, image_dict_train, image_dict_val, image_dict_test


def load_data(vision=None):
    '''
    Create data loaders. 

    Arguments: 
        tabular (bool): Whether to use tabular data
        vision (str): Type of vision encoder 'resnet50', 'dense121' or 'vit' (Default: None --> No images)
    '''
    # Load and pre-process tabular data and labels
    tab_data_train, tab_data_val, tab_data_test, \
        image_dict_train, image_dict_val, image_dict_test = prepare_data()

    # Create datasets
    train_dataset = MultimodalDataset(vision, image_dict_train, tab_data_train)
    val_dataset = MultimodalDataset(vision, image_dict_val, tab_data_val)
    test_dataset = MultimodalDataset(vision, image_dict_test, tab_data_test)

    # Create data loaders
    loader_params = {'batch_size': 32, 'num_workers': 4, 'shuffle': True}
    train_loader = DataLoader(train_dataset, **loader_params)
    val_loader = DataLoader(val_dataset, **loader_params)
    test_loader = DataLoader(test_dataset, **loader_params)

    # Test data loaders
    for pa_image, lateral_image, label_tensor, tabular_tensor in train_loader:
        print(pa_image.shape)
        print(lateral_image.shape)
        print(label_tensor.shape)
        print(tabular_tensor.shape)
        break

    return train_loader, val_loader, test_loader


if __name__ == '__main__': 
    tab_data_train, tab_data_val, tab_data_test, image_dict_train, image_dict_val, image_dict_test = prepare_data()

    # Print the shapes of the dataframes
    print('Tabular data train: ', tab_data_train.shape)
    print('Tabular data val: ', tab_data_val.shape)
    print('Tabular data test: ', tab_data_test.shape)
    print('Image data train: ', len(image_dict_train))
    print('Image data val: ', len(image_dict_val))
    print('Image data test: ', len(image_dict_test))

    # Save the dataframes
    tab_data_train.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_train.csv'), index=False)
    tab_data_val.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_val.csv'), index=False)
    tab_data_test.to_csv(os.path.join(PROCESSED_PATH, 'tab_data_test.csv'), index=False)

    # Save the dictionaries
    np.save(os.path.join(PROCESSED_PATH, 'image_dict_train.npy'), image_dict_train)
    np.save(os.path.join(PROCESSED_PATH, 'image_dict_val.npy'), image_dict_val)
    np.save(os.path.join(PROCESSED_PATH, 'image_dict_test.npy'), image_dict_test)

    # Delete not matched images
    _, image_files, _ = load_images_data()
    for image_file in image_files:
        if image_file not in image_dict_train.keys():
            os.remove(image_file)
        if image_file not in image_dict_val.keys():
            os.remove(image_file)
        if image_file not in image_dict_test.keys():
            os.remove(image_file)

