import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse

from itertools import permutations
from util import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,f1_score
from model import *
from dataset import CustomDataset
from tqdm import tqdm 
# from preprocessing_sensordata import save_path, preprocessed_file_name, save_path_results

def test(lr=1e-3,wd=0,p=0,data=''):
    r"""train model using specific dataset
    Arguments:
        lr (:obj:`float`):
            learning rate for model training
        wd (:obj:`float`):
            weight decay for model optimize
        p (:obj:`float`):
            dropout probablity for model
        data (:obj:`str`):
            select specific dataset
            
    Returns:
        :obj:`tuple`: list of training result. training_set and validation_set 's accuracy,F1 score, MSE
    """
    if data=='':
        #if dataset not selected excute program
        print('No specific data')
        exit()
    
    #load test data
    with open(args.save_path + "/"+ args.preprocessed_file_name + ".pickle.pkl","rb") as fr:
        test = pickle.load(fr)
    
    file_name_list = test[1]

    empty_label_list = [0] * len(test[0])
    
    val_set=CustomDataset(torch.tensor(test[0]).float(), empty_label_list)
    
    #make data loader for model trainig
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #model,optimizer,scheduler and loss functions Declaration
    checkpoint = torch.load('./model_save/pretrained_model.pt')
    Combine_model=CNN2D(p=p).to(device)
    Combine_model.load_state_dict(checkpoint.state_dict())

    # inference for raw dataset
    for _ in tqdm(range(1)):
        #local variable for result check
        class_pred_labels=[]
        count_pred_labels=[]
        
        #validation dataset verify
        Combine_model.eval()
        for _, (x,y) in enumerate(val_dataloader):

            # batch_class, batch_count = y

            if torch.cuda.is_available():
                x = x.to(device)

            pred_count, pred_class = Combine_model(x)
            
            #training step result extend
            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())
            count_pred_labels.extend(pred_count.detach().cpu().numpy())
 
    # save path for result 
    with open(args.save_path_results, 'w', encoding='UTF-8') as f:
        for class_pred, count_pred, file_name in zip(inverse_class(class_pred_labels), np.array(count_pred_labels).round(0).astype(int), file_name_list):
            f.write('{} - [exercise,reps] : {}, {} \n'.format(file_name, class_pred, count_pred))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default='./raw_test_sample/', help='Path where the sensordata file is located')
    parser.add_argument('--save_path', type=str, default='raw_data_torch/', help='Save path for preprocessed torch data')
    parser.add_argument('--num_files', type=int, default='1000', help='Number of files want to preprocess')
    parser.add_argument('--preprocessed_file_name', type=str, default='test', help='Preprocessed file name you want to save')
    parser.add_argument('--save_path_results', type=str, default='./test.txt', help='Save path where the results file')

    args = parser.parse_args()

    print("Preprocessing raw sensordata...")
    save_torch_raw_data(args.dir_path, args.save_path, args.num_files, args.preprocessed_file_name)
    print("Successfully save preprocessed torch data!")

    test(lr=0.0001, data='cat')
    # test(lr=0.0001,data='cat')