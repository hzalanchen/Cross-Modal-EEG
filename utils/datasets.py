import os
import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, image_eeg_pairs, eeg_transform = None):
        self.image_eeg_pairs = image_eeg_pairs
        self.eeg_transform = eeg_transform

    def __getitem__(self, index):
        img, eeg, label = self.image_eeg_pairs[index]
        return img, eeg, label
    
    def __len__(self):
        return len(self.image_eeg_pairs)


class TestDataset(Dataset):
    def __init__(self, test_eeg , eeg_transform = None):
        self.test_eeg = test_eeg
        self.eeg_transform = eeg_transform

    def __getitem__(self, index):
        eeg, label = self.test_eeg[index]
        return eeg, label
    
    def __len__(self):
        return len(self.test_eeg)


def get_train_test_datasets(image_feature_path, eeg_root_path, sub='01', normalize=False):

    train_eeg_path = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_training.npy')
    test_eeg_path = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_test.npy')
    
    train_eeg_data = np.load(train_eeg_path, allow_pickle=True)['preprocessed_eeg_data']
    test_eeg_data = np.load(test_eeg_path, allow_pickle=True)['preprocessed_eeg_data']

    train_eeg_data = np.mean(train_eeg_data, axis=1)
    test_eeg_data = np.mean(test_eeg_data, axis=1)

    if normalize:
        mean = np.mean(train_eeg_data, axis=(0, 2), keepdims=True)
        std = np.std(train_eeg_data, axis=(0, 2), keepdims=True)
        std[std == 0] = 1e-8
        # (X - Mean_train) / Std_train
        train_eeg_data = (train_eeg_data - mean) / std
        test_eeg_data = (test_eeg_data - mean) / std

    train_eeg_tensor = torch.from_numpy(train_eeg_data).float()
    test_eeg_tensor = torch.from_numpy(test_eeg_data).float()

    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']
    test_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_test.pt'))['img_features']


    samples_per_class = 10 
    train_image_eeg_pairs = []

    for sample_index in range(len(train_eeg_tensor)):
        label = sample_index // samples_per_class
        train_image_eeg_pairs.append([
            train_img_feature[sample_index], 
            train_eeg_tensor[sample_index], 
            label
        ])

    test_eeg_pairs = []
    for cls_idx in range(len(test_eeg_tensor)):
        test_eeg_pairs.append([
            test_eeg_tensor[cls_idx], 
            cls_idx
        ])

    return train_image_eeg_pairs, test_img_feature, test_eeg_pairs


def get_train_val_test_datasets_LOSO(image_feature_path, eeg_root_path, sub='01', normalize=True):
    """
    Returns:
        train_image_eeg_pairs: Training pairs (image, eeg, label)
        val_eeg_pairs: Validation pairs (eeg, label)
        test_img_feature: Test image features (visual concept)
        test_eeg_pairs: Test pairs (eeg, label)
    """ 
    # All 10 subjects
    all_subs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    test_sub = sub
    train_subs = [s for s in all_subs if s != test_sub]
    
    samples_per_class = 10
    
    train_eeg_list = []
    for s in train_subs:
        train_eeg_path = os.path.join(eeg_root_path, f'sub-{s}', 'preprocessed_eeg_training.npy')
        eeg_data = np.load(train_eeg_path, allow_pickle=True)['preprocessed_eeg_data']
        eeg_data = np.mean(eeg_data, axis=1) 
        train_eeg_list.append(eeg_data)
    
    val_eeg_list = []
    for s in train_subs:
        val_eeg_path = os.path.join(eeg_root_path, f'sub-{s}', 'preprocessed_eeg_test.npy')
        eeg_data = np.load(val_eeg_path, allow_pickle=True)['preprocessed_eeg_data']
        eeg_data = np.mean(eeg_data, axis=1)  
        val_eeg_list.append(eeg_data)
    
    test_eeg_path = os.path.join(eeg_root_path, f'sub-{test_sub}', 'preprocessed_eeg_test.npy')
    test_eeg_data = np.load(test_eeg_path, allow_pickle=True)['preprocessed_eeg_data']
    test_eeg_data = np.mean(test_eeg_data, axis=1) 
    
    # Concatenate training EEG data for normalization
    train_eeg_concat = np.concatenate(train_eeg_list, axis=0)
    val_eeg_concat = np.concatenate(val_eeg_list, axis=0)

    if normalize:
        mean = np.mean(train_eeg_concat, axis=(0, 2), keepdims=True)
        std = np.std(train_eeg_concat, axis=(0, 2), keepdims=True)
        std[std == 0] = 1e-8
        
        train_eeg_concat = (train_eeg_concat - mean) / std
        val_eeg_concat = (val_eeg_concat - mean) / std
        test_eeg_data = (test_eeg_data - mean) / std
    

    train_eeg_tensor = torch.from_numpy(train_eeg_concat).float()
    val_eeg_tensor = torch.from_numpy(val_eeg_concat).float()
    test_eeg_tensor = torch.from_numpy(test_eeg_data).float()
    
    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']
    test_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_test.pt'))['img_features']
    
    # Repeat train image features 9 times for 9 subjects
    train_img_feature_repeated = train_img_feature.repeat(len(train_subs), 1)
    
    train_image_eeg_pairs = []
    for sample_index in range(len(train_eeg_tensor)):
        # Label is based on the sample index within each subject's data
        label = (sample_index % len(train_img_feature)) // samples_per_class
        train_image_eeg_pairs.append([
            train_img_feature_repeated[sample_index],
            train_eeg_tensor[sample_index],
            label
        ])
    
    val_eeg_pairs = []
    num_val_samples_per_sub = len(val_eeg_tensor) // len(train_subs)
    for sample_index in range(len(val_eeg_tensor)):
        # Label is based on the sample index within each subject's validation data
        label = sample_index % num_val_samples_per_sub
        val_eeg_pairs.append([
            val_eeg_tensor[sample_index],
            label
        ])
    
    test_eeg_pairs = []
    for cls_idx in range(len(test_eeg_tensor)):
        test_eeg_pairs.append([
            test_eeg_tensor[cls_idx],
            cls_idx
        ])
    
    print(f"Test subject: sub-{test_sub}")
    print(f"Train samples: {len(train_image_eeg_pairs)}, Val samples: {len(val_eeg_pairs)}, Test samples: {len(test_eeg_pairs)}")
    
    return train_image_eeg_pairs, val_eeg_pairs, test_img_feature, test_eeg_pairs



# No average the train and test trials repetition axis
def get_image_eeg_pair(image_feature_path, eeg_root_path, sub = '01', total_objs = 1654):
    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_training.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    
    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']

    # get image eeg pair
    image_eeg_pairs = []
    for cls_idx in range(total_objs):
        for count, img_feature in enumerate(train_img_feature[cls_idx*10: cls_idx*10+10]):
            eegs = eeg_datasets[cls_idx*10 + count]
            for i in range(eegs.shape[0]):
                eeg = eegs[i]
                image_eeg_pairs.append([img_feature, eeg , cls_idx])        

    return image_eeg_pairs


def get_test_eeg(image_feature_path, eeg_root_path, sub = '01', total_objs = 200):
    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_test.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']

    test_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_test.pt'))['img_features']

    # get eeg pair
    test_eeg = []
    for cls_idx in range(total_objs):
        eegs = eeg_datasets[cls_idx]
        for j in range(eegs.shape[0]):
            eeg = eegs[j]
            test_eeg.append([eeg, cls_idx])

    return test_img_feature, test_eeg

