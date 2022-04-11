import os
import random
import pandas as pd
import numpy as np
import torch
import pickle

from itertools import permutations
from tqdm import tqdm


def find_class(file):
    if 'squat' in file:
        return 0
    elif 'pushup' in file:
        return 1
    elif 'pullup' in file:
        return 2
    elif 'situp' in file:
        return 3
    elif 'deadlift' in file:
        return 4

def inverse_class(class_labels):
    inverse_class_list = []
    for label in class_labels:
        if label == 0:
            inverse_class_list.append('squat')
        elif label == 1:
            inverse_class_list.append('pushup')
        elif label == 2:
            inverse_class_list.append('pullup')
        elif label == 3:
            inverse_class_list.append('situp')
        elif label == 4:
            inverse_class_list.append('deadlift')

    return inverse_class_list

def fun(file):
    if 'txt' in file:
        return True  

def getfiles(dir_path):
    files = os.listdir(dir_path)
    return list(filter(fun, files))

def convert_16bytes_to_float(array, exponents=10):
    return (np.ascontiguousarray(array).view(dtype=np.int16) / (2 ** exponents)).squeeze()

def decode(data):  # decode raw sensor txt files
    sensordata = data[~data.iloc[:,0].str.startswith("[")].to_numpy().astype(np.float32).astype(np.uint8)

    data_dict = {'accX': convert_16bytes_to_float(sensordata[:,0:2]), 'accY': convert_16bytes_to_float(sensordata[:,2:4]), 'accZ': convert_16bytes_to_float(sensordata[:,4:6]), 'angX': convert_16bytes_to_float(sensordata[:,6:8]), 'angY': convert_16bytes_to_float(sensordata[:,8:10]), 'angZ': convert_16bytes_to_float(sensordata[:,10:12])}
    new_data = pd.DataFrame(data=data_dict)
    new_data['acc_scala'] = (new_data['accY'] ** 2 + new_data['accX'] ** 2 + new_data['accZ'] ** 2).apply(np.sqrt)[:].rolling(3, min_periods=1, center=True).mean()
    new_data['ang_scala'] = (new_data['angY'] ** 2 + new_data['angX'] ** 2 + new_data['angZ'] ** 2).apply(np.sqrt)[:].rolling(3, min_periods=1, center=True).mean()
    
    return new_data

def save_files(data, file_name, save_path):
    data.to_csv(save_path + file_name + '.txt', header=False, index=False)

def sampling(data, sampling_num):
    sampled_data = []
    data_col=data.columns
    data=data.to_numpy()
    for row in range(0, len(data), sampling_num):
        if row + sampling_num > len(data):
            extra_row = row + sampling_num - len(data)
            sampled_data.append(np.mean(data[row:row+extra_row],axis=0))
        else:
            sampled_data.append(np.mean(data[row:row+sampling_num,:],axis=0))        
    return pd.DataFrame(sampled_data,columns=data_col)

def save_torch_raw_data(dir_path, save_path, num, preprocessed_file_name, decoding=True):
    
    max_len = 60 
    files = os.listdir(dir_path)

    max_len_list = []
    file_name_list =[]

    files = getfiles(dir_path)
    random.shuffle(files)
    for file in tqdm(files[:num], desc='make squence'):
        if '.ipynb' in file:
            continue
        
        data = pd.read_csv(dir_path + file, index_col=None, skiprows=[0], header=None)
        
        data = decode(data.astype(str))
        
        # smoothing
        data = data.rolling(10).mean()
        
        # sampling
        data = sampling(data, sampling_num=10)

        # normalization
        data = (data - data.mean())/data.std()  
        if data.isnull().values.any():
            data = data.fillna(0)
        seq_list = []

        if len(data) < max_len * 10:  # if data length over max length, zero padding
            pad = pd.DataFrame(np.zeros((max_len * 10 - len(data), 8)),columns=data.columns)
            cat_df = pd.concat([data, pad], axis=0).reset_index()
            cat_df = cat_df.drop(['index'], axis=1)
            
            for i in range(max_len):  # make sequence data
                seq_list.append(torch.tensor(np.array(cat_df.iloc[i*10 : i*10 + 10,:])))

            max_len_list.append(torch.stack(seq_list))
            file_name_list.append(file)
            
    x_data = torch.stack(max_len_list).float()
    
    # save test data
    with open(save_path + preprocessed_file_name + '.pickle.pkl', 'wb') as f:  
        pickle.dump([x_data, file_name_list], f)

def save_torch(dir_path, save_path, num, preprocessed_file_name, decoding=True):
    
    max_len = 60 
    files = os.listdir(dir_path)

    max_len_list = []
    label = []
    count = []
    
    files = getfiles(dir_path)
    random.shuffle(files)
    for file in tqdm(files[:num], desc='make squence'):
        if '.ipynb' in file:
            continue
        
        data = pd.read_csv(dir_path + file, index_col=None, skiprows=[0], header=None)
        
        if decoding and 'warping' not in file and 'noise' not in file:
            data = decode(data.astype(str))
        
        # smoothing
        data = data.rolling(10).mean()
        
        # sampling
        data = sampling(data, sampling_num=10)

        # normalization
        data = (data - data.mean())/data.std()  
        if data.isnull().values.any():
            data = data.fillna(0)
        seq_list = []

        if len(data) < max_len * 10:  # if data length over max length, zero padding
            pad = pd.DataFrame(np.zeros((max_len * 10 - len(data), 8)),columns=data.columns)
            cat_df = pd.concat([data, pad], axis=0).reset_index()
            cat_df = cat_df.drop(['index'], axis=1)
            
            for i in range(max_len):  # make sequence data
                seq_list.append(torch.tensor(np.array(cat_df.iloc[i*10 : i*10 + 10,:])))

            max_len_list.append(torch.stack(seq_list))
            label.append(torch.tensor(find_class(file)))
            count.append(torch.tensor(int(file.split('_')[1][0])))

    x_data = torch.stack(max_len_list).float()
    y_label = torch.stack(label).float()
    y_count = torch.stack(count).float()
    
    # save test data
    with open(save_path + preprocessed_file_name + '.pickle.pkl', 'wb') as f:  
        pickle.dump([x_data, y_label, y_count], f)

