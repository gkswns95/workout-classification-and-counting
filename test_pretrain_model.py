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

    vaild_combined_label=[]  # (class label, count target)

    #data split. data has 3 dim. each dimention has data,Class label,count target
    for ind in range(len(test[2])):
        vaild_combined_label.append((test[1][ind].item(),test[2][ind].item()))
    
    val_set=CustomDataset(torch.tensor(test[0]).float(), vaild_combined_label)
    
    #make data loader for model trainig
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #model,optimizer,scheduler and loss functions Declaration
    checkpoint = torch.load('./model_save/pretrained_model.pt')
    Combine_model=CNN2D(p=p).to(device)
    Combine_model.load_state_dict(checkpoint.state_dict())
    
    optimizer=torch.optim.Adam(Combine_model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    class_criterion=nn.CrossEntropyLoss()
    count_criterion=nn.MSELoss()

    #traning step
    epochs=1
    #start training
    for epoch in tqdm(range(epochs)):
        #local variable for result check
        class_loss_sum=0
        class_true_labels=[]
        class_pred_labels=[]
        
        count_loss_sum=0
        count_true_labels=[]
        count_pred_labels=[]
        
        #validation dataset verify
        Combine_model.eval()
        for e_num,(x,y) in enumerate(val_dataloader):

            batch_class, batch_count = y

            x,batch_class, batch_count = x.to(device),batch_class.long().to(device),batch_count.float().to(device)
            pred_count, pred_class = Combine_model(x)
            
            #training step result extend
            class_true_labels.extend(batch_class.cpu().numpy())
            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())
            
            #calculate loss, not for train just for check result 
            class_loss = class_criterion(pred_class,batch_class)
            class_loss_sum += class_loss.detach().item()  
            
            #training step result extend
            count_true_labels.extend(batch_count.cpu().numpy())
            count_pred_labels.extend(pred_count.detach().cpu().numpy())
            
            #calculate loss, not for train just for check result 
            count_loss=count_criterion(pred_count,batch_count)
            count_loss_sum+=count_loss.detach().item()        

        #calculate validation step result
        mse= mean_squared_error(count_true_labels, count_pred_labels)
        acc=accuracy_score(class_true_labels, class_pred_labels)
        
        if e_num == 0:
            e_num = 1

        print(f'validataion \t  class loss mean {round(class_loss_sum/e_num,3)} acc :{round(acc,5)} count loss mean {round(count_loss_sum/e_num,3)}  MSE :{round(mse,3)}',end='\n\n')

        scheduler.step()

    #trainig end 

    #result record
    count_true_labels=[]
    count_pred_labels=[]

    class_true_labels=[]
    class_pred_labels=[]

    Combine_model.eval()

    file_name ='./txt_save/'
    file_name += f'data-{data}_lr-{lr}_wd-{wd}'

    #reset variable
    count_true_labels=[]
    count_pred_labels=[]

    class_true_labels=[]
    class_pred_labels=[]

    #get validation result
    for e_num,(x,y) in enumerate(val_dataloader):
        batch_class,batch_count=y
        x,batch_class,batch_count=x.to(device),batch_class.long().to(device),batch_count.float().to(device)

        pred_count,pred_class=Combine_model(x)

        class_true_labels.extend(batch_class.cpu().numpy())
        class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())

        count_true_labels.extend(batch_count.cpu().numpy())
        count_pred_labels.extend(pred_count.detach().cpu().numpy())

    v_acc=accuracy_score(class_true_labels,class_pred_labels)
    v_mse=mean_squared_error(count_true_labels,count_pred_labels)
    v_f1=f1_score(list(class_true_labels), class_pred_labels, average='macro')

    count_acc = accuracy_score(np.array(count_true_labels).astype(int), np.array(count_pred_labels).round(0).astype(int))
    # print result
    print('valid class acc: ', v_acc)
    print('valid class F1: ', v_f1)
    print('valid count mse: ', v_mse)
    print("valid accuracy count : {}%".format(count_acc * 100))
    
    # save path for result 
    with open(args.save_path_results, 'w', encoding='UTF-8') as f:
        for class_pred, count_pred in zip(inverse_class(class_pred_labels), np.array(count_pred_labels).round(0).astype(int)):
            f.write('[exercise,reps] : {}, {} \n'.format(class_pred, count_pred))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default='./raw_test/', help='Path where the sensordata file is located')
    parser.add_argument('--save_path', type=str, default='test_torch/', help='Save path for preprocessed torch data')
    parser.add_argument('--num_files', type=int, default='1000', help='Number of files want to preprocess')
    parser.add_argument('--preprocessed_file_name', type=str, default='test', help='Preprocessed file name you want to save')
    parser.add_argument('--save_path_results', type=str, default='./test.txt', help='Save path where the results file')

    args = parser.parse_args()

    print("Preprocessing raw sensordata...")
    save_torch(args.dir_path, args.save_path, args.num_files, args.preprocessed_file_name)
    print("Successfully save preprocessed torch data!")

    test(lr=0.0001, data='cat')
    # test(lr=0.0001,data='cat')