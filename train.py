import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,f1_score
import pickle
from model import *
from dataset import CustomDataset

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(lr=1e-3,wd=0,p=0,data=''):
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
    
    dir_path = 'train_torch'
    #load train data
    if data=='noise':
        with open(dir_path + "/noise.pickle.pkl","rb") as fr:
            train = pickle.load(fr)
    elif data=='split':
        with open(dir_path + "/split.pickle.pkl","rb") as fr:
            train = pickle.load(fr)
    elif data=='time':            
        with open(dir_path + "/warping.pickle.pkl","rb") as fr:
            train = pickle.load(fr)
    elif data=='raw':            
        with open(dir_path + "/raw.pickle.pkl","rb") as fr:
            train = pickle.load(fr)
    elif data=='cat':
        with open(dir_path + "/all.pickle.pkl","rb") as fr:
            train = pickle.load(fr)

    #load test data
    with open("test_torch/test.pickle.pkl","rb") as fr:
        test = pickle.load(fr)

    train_combined_label=[]
    vaild_combined_label=[]

    #data split. data has 3 dim. each dimention has data,Class label,count target
    for ind in range(len(train[2])):
        train_combined_label.append((train[1][ind].item(),train[2][ind].item()))

    for ind in range(len(test[2])):
        vaild_combined_label.append((test[1][ind].item(),test[2][ind].item()))
        
    train_set=CustomDataset(torch.tensor(train[0]).float(), train_combined_label)
    val_set=CustomDataset(torch.tensor(test[0]).float(), vaild_combined_label)
    
    #make data loader for model trainig
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    
    #tensorboard setting
    eventid = datetime.now().strftime('runs/capstone-%Y%m-%d%H-%M%S-')
    eventid+=f'data - {data} lr-{lr} wd - {wd}'
    writer = SummaryWriter(eventid)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #model,optimizer,scheduler and loss functions Declaration
    # checkpoint = torch.load('./model_save/saved_model.pt')
    Combine_model=CNN2D(p=p).to(device)
    # Combine_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer=torch.optim.Adam(Combine_model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    class_criterion=nn.CrossEntropyLoss()
    count_criterion=nn.MSELoss()

    #traning step
    epochs=60
    #start training
    for epoch in tqdm(range(epochs)):

        #local variable for result check
        class_loss_sum=0
        class_true_labels=[]
        class_pred_labels=[]
        
        count_loss_sum=0
        count_true_labels=[]
        count_pred_labels=[]

        #training step
        Combine_model.train()
        
        for e_num,(x,y) in enumerate(train_dataloader):
            batch_class,batch_count=y
            
            x,batch_class,batch_count=x.to(device),batch_class.long().to(device),batch_count.float().to(device)#data to device
            
            Combine_model.zero_grad()
            
            pred_count,pred_class=Combine_model(x)
            
            #training step result extend
            class_true_labels.extend(batch_class.cpu().numpy())
            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())

            #calculate loss
            class_loss=class_criterion(pred_class,batch_class)
            #loss result append
            class_loss_sum+=class_loss.detach().item()
            
            #training step result extend
            count_true_labels.extend(batch_count.cpu().numpy())
            count_pred_labels.extend(pred_count.detach().cpu().numpy())

            #calculate loss
            count_loss=count_criterion(pred_count,batch_count)
            #loss result append
            count_loss_sum+=count_loss.detach().item()   
            
            #combine loss for gradient calculate and update model parameter
            total_loss=count_loss+class_loss
            total_loss.backward()
            optimizer.step()

        #calculate training step result            
        mse= mean_squared_error(count_true_labels,count_pred_labels)    
        acc=accuracy_score(class_true_labels,class_pred_labels)
        
        print(f'train \t\t class loss mean {round(class_loss_sum/e_num,3)} acc :{round(acc,5)} count loss mean {round(count_loss_sum/e_num,3)}  MSE :{round(mse,3)}')
        writer.add_scalar('Train MSE', round(mse,3), epoch)
        writer.add_scalar('Train Accuracy', round(acc,5), epoch)
        
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

            batch_class,batch_count=y
            x,batch_class,batch_count=x.to(device),batch_class.long().to(device),batch_count.float().to(device)
            pred_count,pred_class=Combine_model(x)
            
            #training step result extend
            class_true_labels.extend(batch_class.cpu().numpy())
            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())
            
            #calculate loss, not for train just for check result 
            class_loss=class_criterion(pred_class,batch_class)
            class_loss_sum+=class_loss.detach().item()  
            
            #training step result extend
            count_true_labels.extend(batch_count.cpu().numpy())
            count_pred_labels.extend(pred_count.detach().cpu().numpy())
            
            #calculate loss, not for train just for check result 
            count_loss=count_criterion(pred_count,batch_count)
            count_loss_sum+=count_loss.detach().item()        
       #calculate validation step result    
        mse= mean_squared_error(count_true_labels,count_pred_labels)
        acc=accuracy_score(class_true_labels,class_pred_labels)
        
        print(f'validataion \t  class loss mean {round(class_loss_sum/e_num,3)} acc :{round(acc,5)} count loss mean {round(count_loss_sum/e_num,3)}  MSE :{round(mse,3)}',end='\n\n')
        writer.add_scalar('validaion MSE', round(mse,3), epoch)
        writer.add_scalar('validaion Accuracy', round(acc,5), epoch)
        scheduler.step()
    #trainig end 

    #result record
    count_true_labels=[]
    count_pred_labels=[]

    class_true_labels=[]
    class_pred_labels=[]

    Combine_model.eval()

    file_name='./txt_save/'
    file_name+=f'data-{data}_lr-{lr}_wd-{wd}'

    #generate txt for result
    f=open(file_name+'.txt','w')
    f.write(f'-------------data - {data}_ lr-{lr} wd - {wd}-----------\n')

    #get train result
    for e_num,(x,y) in enumerate(train_dataloader):
        batch_class,batch_count=y
        x,batch_class,batch_count=x.to(device),batch_class.long().to(device),batch_count.float().to(device)

        pred_count,pred_class=Combine_model(x)

        class_true_labels.extend(batch_class.cpu().numpy())
        class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())

        count_true_labels.extend(batch_count.cpu().numpy())
        count_pred_labels.extend(pred_count.detach().cpu().numpy())

    t_acc=accuracy_score(class_true_labels,class_pred_labels)
    t_mse=mean_squared_error(count_true_labels,count_pred_labels)
    t_f1=f1_score(list(class_true_labels), class_pred_labels, average='macro')

    # print result and txt file write
    print('train class acc: ',t_acc)
    f.write('train class acc: ')
    f.write(str(t_acc)+'\n')

    print('train class F1: ',t_f1)
    f.write('train class F1 : ')
    f.write(str(t_f1)+'\n')
    

    print('train count acc: ',t_mse)
    f.write('train count acc: ')
    f.write(str(t_mse)+'\n')

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

    # print result and txt file write
    print('valid class acc: ',v_acc)
    f.write('valid class acc: ')
    f.write(str(v_acc)+'\n')
    
    print('valid class F1: ',v_f1)
    f.write('valid class F1 : ')
    f.write(str(v_f1)+'\n')
    
    print('valid count acc: ',v_mse)
    f.write('valid count acc: ')
    f.write(str(v_mse)+'\n')

    #trained model save
    model_name=f'saved_model'
    torch.save(Combine_model, 'model_save/'+model_name+'.pt')
    
    return t_acc,t_f1,t_mse,v_acc,v_f1,v_mse

if __name__ == "__main__":

    train(lr=0.0001,data='cat')