import pandas as pd
import os
import numpy as np
seed = 42
np.random.seed(seed)

from data import *

MIMIC_PATH = '../data/mimic-iv/'
PATH_ADMISSIONS = MIMIC_PATH + 'admissions.csv.gz'
PATH_PATIENTS = MIMIC_PATH + 'patients.csv.gz'
PATH_SERVICES = MIMIC_PATH + 'services.csv.gz'
PATH_METADATA = '../data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz'
PATH_LABELS = '../data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv'

PATH_PREPROCESSED_DATA = '../data/processed_data/tab_data_total.csv'
PATH_DATA_TRAIN = '../data/processed_data/tab_data_train.csv'
PATH_DATA_VAL = '../data/processed_data/tab_data_val.csv'
PATH_DATA_TEST = '../data/processed_data/tab_data_test.csv'
PATH_LABELS_TRAIN = '../data/processed_data/labels_train.csv'
PATH_LABELS_VAL = '../data/processed_data/labels_val.csv'
PATH_LABELS_TEST = '../data/processed_data/labels_test.csv'

PATH_IMAGES = '../data/mimic-cxr/'

def preprocess_data():
    # check if preprocessed data already exists
    if not os.path.exists(PATH_PREPROCESSED_DATA):
        # load data
        admissions, patients, services, metadata = load_tabular_data()

        #----------------------------------- PREPROCESS ADMISSIONS AND PATIENTS -----------------------------------#

        admissions_cleaned = admissions.drop(columns=[
            'dischtime', 'deathtime', 'discharge_location', 'edregtime', 'edouttime', 'hospital_expire_flag'])
        patients_cleaned = patients.drop(columns=['dod'])

        # Merge the two tables
        tabular_data = pd.merge(admissions_cleaned, patients_cleaned, on='subject_id', how='left')

        # The admittime has the format YYYY-MM-DD HH:MM:SS, convert it to DateTime
        tabular_data['admittime'] = pd.to_datetime(tabular_data['admittime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


        #----------------------------------- PREPROCESS METADATA OF IMAGES -----------------------------------#

        len_old = len(metadata)
        # Keep only the study ideas with exactly 2 images
        groups = metadata.groupby('study_id')
        study_ids = []
        for study_id, group in groups:
            if len(group) > 1:
                # Keep study_id if there is a ViewPosition PA and LATERAL
                if 'PA' in group['ViewPosition'].values and 'LATERAL' in group['ViewPosition'].values:
                    study_ids.append(study_id)
        metadata = metadata[metadata['study_id'].isin(study_ids)]
        # Keep only the images with ViewPosition PA and LATERAL
        metadata = metadata[metadata['ViewPosition'].isin(['PA', 'LATERAL'])]
        # Group by study_id and by ViewPosition and keep only the first image
        metadata = metadata.groupby(['study_id', 'ViewPosition']).first().reset_index()
        len_new = len(metadata)
        print(f'Number of studies with Lateral and PA image: {len_new} ({len_new/len_old*100:.2f}%)')

        #----------------------------------- KEEP STUDIES WHERE THE TIME OF STUDY IS KNOWN-----------------------------------#

        metadata_time = pd.DataFrame(columns=['subject_id', 'StudyDate', 'StudyTime', 'dicom_id', 'study_id', 'ViewPosition'])
        # We only keep the columns subject_id, StudyDate, StudyTime, dicom_id, study_id
        metadata_time['dicom_id'] = metadata['dicom_id']
        metadata_time['study_id'] = metadata['study_id']
        metadata_time['ViewPosition'] = metadata['ViewPosition']
        metadata_time['subject_id'] = metadata['subject_id']
        # The study data is in the form YYYYMMDD, we need to convert it to DateTime
        metadata_time['StudyDate'] = pd.to_datetime(metadata['StudyDate'], format='%Y%m%d').dt.date
        # Studytime is in the form HHMMSS.SSS or HHMMSS or HHMMSS.SS or HHMMSS.S, remove the fraction of seconds, round to the nearest second, it is a numpy float
        metadata_time['StudyTime'] = metadata['StudyTime'].astype(str).str.split('.').str[0]
        metadata_time['StudyTime'] = pd.to_datetime(metadata['StudyTime'], format='%H%M%S', errors='coerce').dt.time
        old_len = len(metadata_time)
        #Drop rows where either StudyDate or StudyTime is NaT
        metadata_time = metadata_time.dropna(subset=['StudyDate', 'StudyTime'])
        new_len = len(metadata_time)
        print(f'Dropped {old_len - new_len} rows because the date or time of the study was not known')
        # Now combine the date and time into one column as DateTime
        metadata_time['StudyDateTime'] = pd.to_datetime(metadata_time['StudyDate'].astype(str) + ' ' + metadata_time['StudyTime'].astype(str))
        # Drop the columns StudyDate and StudyTime
        metadata_time = metadata_time.drop(columns=['StudyDate', 'StudyTime'])

        # ---------------------------------------- MERGE METADATA WITH ADMISSION/PATIENT DATA ----------------------------------------#

        # We first merge the two tables
        merged = pd.merge(metadata_time, tabular_data, on='subject_id', how='left')
        # We keep only the rows where the StudyDateTime is after the admittime
        merged = merged[merged['StudyDateTime'] >= merged['admittime']]
        # We group by dicom_id and keep the row with the latest admittime
        merged = merged.sort_values('admittime').groupby('dicom_id').last().reset_index()



        # ---------------------------------------- ADD SERVICES TO THE DATA ----------------------------------------#

        unique_services = services['curr_service'].unique()
        services['transfertime'] = pd.to_datetime(services['transfertime'], format='%Y-%m-%d %H:%M:%S').dt.date

        # Merge the services with the tabular data, on hadm_id and subject_id
        tabular_metadata_services = pd.merge(merged, services, on=['hadm_id', 'subject_id'], how='left')

        # Now only keep the rows where the transfertime is before the StudyDateTime
        tabular_metadata_services = tabular_metadata_services[tabular_metadata_services['transfertime'] <= tabular_metadata_services['StudyDateTime']]

        # Group by dicom_id and take the row with the latest transfertime
        tabular_metadata_services = tabular_metadata_services.sort_values('transfertime').groupby('dicom_id').last().reset_index()

        for service in unique_services:
            #initiate the column with 0
            tabular_metadata_services[service] = 0

        # Set the service to 1 if it is in the curr_service column
        for index, row in tabular_metadata_services.iterrows():
            tabular_metadata_services.loc[index, row['curr_service']] = 1

        # Drop the curr_service column, the transfertime column and the prev_service column

        tabular_metadata_services = tabular_metadata_services.drop(columns=['curr_service', 'transfertime', 'prev_service'])
        
        print('Number of pictures (front and lateral together): ', len(merged))
        print('Number of unique subjects: ', len(merged['subject_id'].unique()))
        print('Number of unique studies: ', len(merged['study_id'].unique()))

        tabular_metadata_services = tabular_metadata_services[tabular_metadata_services['ViewPosition']=='PA']

        # Remove column dicom_id, ViewPosition, StudyDateTime, hadm_id, admit_provider_id
        tabular_metadata_services_prep = tabular_metadata_services.drop(columns=['dicom_id', 'ViewPosition', 'StudyDateTime', 'hadm_id', 'admit_provider_id', 'anchor_year', 'admittime'])

        # One hot encode admission_type, admission_location, language, marital_status, race, gender, anchor_year_group, anchor_year_group
        tabular_metadata_services_prep = pd.get_dummies(tabular_metadata_services_prep, columns=['admission_type', 'admission_location', 'language', 'marital_status', 'race', 'gender', 'anchor_year_group', 'anchor_year_group', 'insurance'])

        # save the preprocessed data
        tabular_metadata_services_prep.to_csv(PATH_PREPROCESSED_DATA, index=False)
        return tabular_metadata_services_prep
    else:
        return pd.read_csv(PATH_PREPROCESSED_DATA)
    
def split_data(data, train_size=0.75, val_size=0.1, test_size=0.15):
    if not os.path.exists(PATH_DATA_TRAIN) or not os.path.exists(PATH_DATA_VAL) \
        or not os.path.exists(PATH_DATA_TEST) or not os.path.exists(PATH_LABELS_TRAIN) \
            or not os.path.exists(PATH_LABELS_VAL) or not os.path.exists(PATH_LABELS_TEST):
        PATH_LABEL = '../data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv'
        labels = pd.read_csv(PATH_LABEL)
        tab_data = preprocess_data()
        labels = labels.replace(1, 2)   # Positive: 1 -> 2
        labels = labels.replace(-1, 1)  # Uncertain: -1 -> 1
        labels = labels.fillna(0)       # Missing == Negative: NaN -> 0

        # Get all the unique study_ids
        study_ids = tab_data['study_id'].unique()
        # Shuffle the study_ids
        np.random.shuffle(study_ids)
        # Get the number of study_ids
        num_study_ids = len(study_ids)
        # Get the number of study_ids in the validation set
        num_val = int(num_study_ids * val_size)
        # Get the number of study_ids in the test set
        num_test = int(num_study_ids * test_size)

        # Split the study_ids into train, val, and test sets
        study_ids_train = study_ids[num_val + num_test:]
        study_ids_val = study_ids[:num_val]
        study_ids_test = study_ids[num_val:num_val + num_test]

        # Get the tabular data for the train, val, and test sets
        tab_data_train = tab_data[tab_data['study_id'].isin(study_ids_train)]
        tab_data_val = tab_data[tab_data['study_id'].isin(study_ids_val)]
        tab_data_test = tab_data[tab_data['study_id'].isin(study_ids_test)]

        labels_train = labels[labels['study_id'].isin(study_ids_train)]
        labels_val = labels[labels['study_id'].isin(study_ids_val)]
        labels_test = labels[labels['study_id'].isin(study_ids_test)]

        # Save the train, val, and test sets
        tab_data_train.to_csv('../data/processed_data/tab_data_train.csv', index=False)
        tab_data_val.to_csv('../data/processed_data/tab_data_val.csv', index=False)
        tab_data_test.to_csv('../data/processed_data/tab_data_test.csv', index=False)

        labels_train.to_csv('../data/processed_data/labels_train.csv', index=False)
        labels_val.to_csv('../data/processed_data/labels_val.csv', index=False)
        labels_test.to_csv('../data/processed_data/labels_test.csv', index=False)

        # Check proportions of total, train, val, and test sets
        print(f'Total: {len(tab_data)}\nTrain: {len(tab_data_train)/len(tab_data)}%\
              \nVal: {len(tab_data_val)/len(tab_data)}%\nTest: {len(tab_data_test)/len(tab_data)}%')

    else:   
        tab_data_train = pd.read_csv(PATH_DATA_TRAIN)
        tab_data_val = pd.read_csv(PATH_DATA_VAL)
        tab_data_test = pd.read_csv(PATH_DATA_TEST)

        labels_train = pd.read_csv(PATH_LABELS_TRAIN)
        labels_val = pd.read_csv(PATH_LABELS_VAL)
        labels_test = pd.read_csv(PATH_LABELS_TEST)

        print('Total set: ', len(tab_data_train) + len(tab_data_val) + len(tab_data_test))
        print('Percent train: ', len(tab_data_train) / (len(tab_data_train) + len(tab_data_val) + len(tab_data_test)))
        print('Percent val: ', len(tab_data_val) / (len(tab_data_train) + len(tab_data_val) + len(tab_data_test)))
        print('Percent test: ', len(tab_data_test) / (len(tab_data_train) + len(tab_data_val) + len(tab_data_test)))

    return tab_data_train, tab_data_val, tab_data_test, labels_train, labels_val, labels_test


def load_tabular_data():
    '''
    Load admissions, patients, and icustays
    '''
    admissions = pd.read_csv(MIMIC_PATH+'admissions.csv.gz')
    patients = pd.read_csv(MIMIC_PATH+'patients.csv.gz')
    services = pd.read_csv(MIMIC_PATH+'services.csv.gz')
    metadata = pd.read_csv(PATH_METADATA)
    return admissions, patients, services, metadata

def load_images_data():
    labels_data = pd.read_csv(PATH_IMAGES + 'mimic-cxr-2.0.0-chexpert.csv')
    image_files = list_images(PATH_IMAGES + 'files')
    info_jpg = pd.read_csv(PATH_IMAGES + 'mimic-cxr-2.0.0-metadata.csv')
    return labels_data, image_files, info_jpg

def filter_images(labels_data, image_files, info_jpg, tab_data):

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