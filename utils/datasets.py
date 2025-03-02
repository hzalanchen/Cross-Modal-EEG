import os
import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import Dataset

def preprocess_eeg(eeg, eeg_transform=None):
    eeg = torch.from_numpy(eeg).float()
    eeg = eeg.unsqueeze(0)
    if eeg_transform:
        eeg = eeg_transform(eeg)
    return eeg


class TrainDataset(Dataset):
    def __init__(self, image_eeg_pairs, eeg_transform = None):
        self.image_eeg_pairs = image_eeg_pairs
        self.eeg_transform = eeg_transform

    def __getitem__(self, index):
        img, eeg, label = self.image_eeg_pairs[index]
        eeg = preprocess_eeg(eeg, self.eeg_transform)
        return img, eeg, label
    
    def __len__(self):
        return len(self.image_eeg_pairs)


class TestDataset(Dataset):
    def __init__(self, test_eeg , eeg_transform = None):
        self.test_eeg = test_eeg
        self.eeg_transform = eeg_transform

    def __getitem__(self, index):
        eeg, label = self.test_eeg[index]
        eeg = preprocess_eeg(eeg, self.eeg_transform)
        return eeg, label
    
    def __len__(self):
        return len(self.test_eeg)


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


# Average the train and test trials repetition axis
def get_image_eeg_pair_Rep(image_feature_path, eeg_root_path, sub = '01', total_objs = 1654):
    samples_per_class = 10
    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_training.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    eeg_datasets = np.mean(eeg_datasets, axis = 1)

    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']
    image_eeg_pairs = []
    for sample_index in range(len(eeg_datasets)):
        label = sample_index // samples_per_class
        image_eeg_pairs.append([train_img_feature[sample_index], eeg_datasets[sample_index], label])
        
    return image_eeg_pairs


def get_test_eeg_Rep(image_feature_path, eeg_root_path, sub = '01', total_objs = 200):
    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_test.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    eeg_datasets = np.mean(eeg_datasets, axis = 1)

    # Huggingface/CLIP-ViT-H-14
    test_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_test.pt'))['img_features']
    # print(test_img_feature.shape)

    test_eeg = []
    for cls_idx in range(total_objs):
        test_eeg.append([eeg_datasets[cls_idx], cls_idx])
    return test_img_feature, test_eeg


def get_train_val(image_feature_path, eeg_root_path, sub = '01', total_objs = 1654, val_nums = 50, val_class = True):
    np.random.seed(20244)
    if not val_class:
        val_class = np.random.choice(total_objs, val_nums, replace = False)
        val_class.sort()
        np.save("./utils/val_class_index_seed20244.npy", val_class)
    else:
        val_class = np.load('./utils/val_class_index_seed20244.npy')
        val_class.sort()
    # print(val_class)
    train_class = np.setdiff1d(np.arange(total_objs), val_class)
    train_class.sort()
    # print(train_class)
    val_class_map = {idx: new_idx for new_idx, idx in enumerate(val_class)}
    train_class_map = {idx: new_idx for new_idx, idx in enumerate(train_class)}

    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_training.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']

    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']

    image_eeg_pairs_train = []
    image_eeg_pairs_val = []
    image_feature_avg_val = []
    for cls_idx in range(total_objs):
        temp_feature_list = []
        for count, img_feature in enumerate(train_img_feature[cls_idx*10: cls_idx*10+10]):
            eegs = eeg_datasets[cls_idx*10 + count]
            if cls_idx in val_class:
                temp_feature_list.append(img_feature)
                for i in range(eegs.shape[0]):
                    eeg = eegs[i]
                    image_eeg_pairs_val.append([img_feature, eeg , val_class_map[cls_idx]]) 
            else:
                for i in range(eegs.shape[0]):
                    eeg = eegs[i]
                    image_eeg_pairs_train.append([img_feature, eeg , train_class_map[cls_idx]])
        if temp_feature_list:
            combined_tensor = torch.stack(temp_feature_list)
            avg_array = combined_tensor.mean(dim=0)
            image_feature_avg_val.append(avg_array)
    val_image_feature = torch.stack(image_feature_avg_val)

    return image_eeg_pairs_train, image_eeg_pairs_val, train_class_map, val_class_map, val_image_feature


def get_train_val_Rep(image_feature_path, eeg_root_path, sub = '01', total_objs = 1654, val_nums = 50, val_class = True):
    np.random.seed(20244)
    if not val_class:
        val_class = np.random.choice(total_objs, val_nums, replace = False)
        val_class.sort()
        np.save("./utils/val_class_index_seed20244.npy", val_class)
    else:
        val_class = np.load('./utils/val_class_index_seed20244.npy')
        val_class.sort()
    # print(val_class)
    train_class = np.setdiff1d(np.arange(total_objs), val_class)
    train_class.sort()
    # print(train_class)
    val_class_map = {idx: new_idx for new_idx, idx in enumerate(val_class)}
    train_class_map = {idx: new_idx for new_idx, idx in enumerate(train_class)}

    # dict key = ['preprocessed_eeg_data', 'ch_names', 'times']
    eeg_files = os.path.join(eeg_root_path, f'sub-{sub}', 'preprocessed_eeg_training.npy')
    eeg_datasets = np.load(eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    eeg_datasets = np.mean(eeg_datasets, axis = 1)

    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path, 'ViT-H-14_features_train.pt'))['img_features']

    image_eeg_pairs_train = []
    image_eeg_pairs_val = []
    image_feature_avg_val = []
    for cls_idx in range(total_objs):
        temp_feature_list = []
        for count, img_feature in enumerate(train_img_feature[cls_idx*10: cls_idx*10+10]):
            eeg = eeg_datasets[cls_idx*10 + count]
            if cls_idx in val_class:
                temp_feature_list.append(img_feature)
                image_eeg_pairs_val.append([img_feature, eeg , val_class_map[cls_idx]]) 
            else:
                image_eeg_pairs_train.append([img_feature, eeg , train_class_map[cls_idx]])
        if temp_feature_list:
            combined_tensor = torch.stack(temp_feature_list)
            avg_array = combined_tensor.mean(dim=0)
            image_feature_avg_val.append(avg_array)
    val_image_feature = torch.stack(image_feature_avg_val)

    return image_eeg_pairs_train, image_eeg_pairs_val, train_class_map, val_class_map, val_image_feature


def leave_one_for_test_Rep(image_feature_path, eeg_root_path, exclude_sub = '01', total_objs_test = 200):
    sample_per_class = 10
    total_sub = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    exclude_sub = f'sub-{exclude_sub}'
    total_sub.remove(exclude_sub)
    
    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path[0], 'ViT-H-14_features_train.pt'))['img_features']
    test_img_feature = torch.load(os.path.join(image_feature_path[1], 'ViT-H-14_features_test.pt'))['img_features']

    train_eegs = []
    train_img_eeg_pairs = []
    for sub in total_sub:
        eeg_file = os.path.join(eeg_root_path, sub, 'preprocessed_eeg_training.npy')
        eeg_dataset = np.load(eeg_file, allow_pickle=True)['preprocessed_eeg_data']
        eeg_dataset = np.mean(eeg_dataset, axis = 1)
        train_eegs.append(eeg_dataset)
    
    for sub_dataset in train_eegs:
        for sample_index in range(len(sub_dataset)):
            label = sample_index // sample_per_class
            train_img_eeg_pairs.append([train_img_feature[sample_index], sub_dataset[sample_index], label])

    
    test_eeg_files = os.path.join(eeg_root_path, exclude_sub, 'preprocessed_eeg_test.npy')
    test_eeg_dataset = np.load(test_eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    test_eeg_dataset = np.mean(test_eeg_dataset, axis = 1)

    test_eeg = []
    for cls_idx in range(total_objs_test):
        test_eeg.append([test_eeg_dataset[cls_idx], cls_idx])
    print(f"Exclude_sub is {exclude_sub}, total_train_sample is {len(train_img_eeg_pairs)}, total_test_sample is {len(test_eeg)}")

    return train_img_eeg_pairs, test_img_feature, test_eeg
        
        
def leave_one_for_test(image_feature_path, eeg_root_path, exclude_sub = '01', total_objs_train = 1654, total_objs_test = 200):
    sample_per_class = 10
    total_sub = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    exclude_sub = f'sub-{exclude_sub}'
    total_sub.remove(exclude_sub)
    
    # Huggingface/CLIP-ViT-H-14
    train_img_feature = torch.load(os.path.join(image_feature_path[0], 'ViT-H-14_features_train.pt'))['img_features']
    test_img_feature = torch.load(os.path.join(image_feature_path[1], 'ViT-H-14_features_test.pt'))['img_features']

    train_eegs = []
    train_img_eeg_pairs = []
    for sub in total_sub:
        eeg_file = os.path.join(eeg_root_path, sub, 'preprocessed_eeg_training.npy')
        eeg_dataset = np.load(eeg_file, allow_pickle=True)['preprocessed_eeg_data']
        # eeg_dataset = np.mean(eeg_dataset, axis = 1)
        train_eegs.append(eeg_dataset)
    
    for sub_dataset in train_eegs:
        for cls_idx in range(total_objs_train):
            for count, img_feature in enumerate(train_img_feature[cls_idx*10: cls_idx*10+10]):
                eegs = sub_dataset[cls_idx*10 + count]
                for i in range(eegs.shape[0]):
                    eeg = eegs[i]
                    train_img_eeg_pairs.append([img_feature, eeg , cls_idx])  
    
    test_eeg_files = os.path.join(eeg_root_path, exclude_sub, 'preprocessed_eeg_test.npy')
    test_eeg_dataset = np.load(test_eeg_files, allow_pickle=True)['preprocessed_eeg_data']
    test_eeg_dataset = np.mean(test_eeg_dataset, axis = 1)

    test_eeg = []
    for cls_idx in range(total_objs_test):
        test_eeg.append([test_eeg_dataset[cls_idx], cls_idx])
    print(f"Exclude_sub is {exclude_sub}, total_train_sample is {len(train_img_eeg_pairs)}, total_test_sample is {len(test_eeg)}")

    return train_img_eeg_pairs, test_img_feature, test_eeg
        
    
    

# train_img_eeg_pairs, test_img_feature, test_eeg = leave_one_for_test_Rep(['/chz/data/Things-EEG/train_feature/', '/chz/data/Things-EEG/center_feature/'] , '/chz/data/Things-EEG/Preprocessed_data_250Hz', exclude_sub=format(1, '02'))
# print(len(train_img_eeg_pairs), len(test_eeg))
# print(test_img_feature.shape)
# print(train_img_eeg_pairs[0][2],train_img_eeg_pairs[9][2],train_img_eeg_pairs[10][2],train_img_eeg_pairs[16540][2],train_img_eeg_pairs[16550][2])
# train_set = set()
# val_set = set()
# for a,b,c in image_eeg_pairs_train:
#     train_set.add(c)
# print(train_set)
# for a,b,c in image_eeg_pairs_val:
#     val_set.add(c)
# print(val_set)

