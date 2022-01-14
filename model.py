import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self,in_channel,hidden_channel=None,out_channel=None,kernel=None):

        r"""residual strucure block
        Arguments:
            in_channel (:obj:`int`):
                input data channel
            hidden_channel (:obj:`int`):
                hidden channel for hidden layer in residual block
            out_channel (:obj:`int`):
                residual block's output channel
            kernel (:obj:'int'):
                kernel size for convolution network
        """
        super(residual_block,self).__init__()

        if hidden_channel==None:
            hidden_channel=in_channel

        if out_channel == None:
            out_channel=in_channel

        if kernel==None:
            kernel=3
            
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=hidden_channel,kernel_size=kernel,padding=(kernel[0]//2,kernel[1]//2))
        self.conv2=nn.Conv2d(in_channels=hidden_channel,out_channels=hidden_channel,kernel_size=kernel,padding=(kernel[0]//2,kernel[1]//2))
        
        self.activation=nn.ReLU()

    def forward(self,x):
        shortcut = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))+shortcut         
        
        return x

class CNN2D(nn.Module):
    def __init__(self,p):
        super(CNN2D,self).__init__()
        
        self.hidden_dim = 256

        #encoder layer. all layer with convolution network
        self.cnn_1=nn.Conv2d(in_channels=8,out_channels=self.hidden_dim,kernel_size=1,padding=0)
        self.resi1=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi2=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi3=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi4=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi5=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi6=residual_block(self.hidden_dim,kernel=(5, 5))    
        self.resi7=residual_block(self.hidden_dim,kernel=(5, 5))
        self.resi8=residual_block(self.hidden_dim,kernel=(5, 5))

        #pooling layer for encoder 
        self.pool1=nn.MaxPool2d((1, 2))
        self.pool2=nn.MaxPool2d((1, 2))
        self.pool3=nn.MaxPool2d((1, 2))
        self.pool4=nn.MaxPool2d((2, 1))
        self.pool5=nn.MaxPool2d((2, 1))
        self.pool6=nn.MaxPool2d((2, 1))
        self.pool7=nn.MaxPool2d((2, 1))
        self.pool8=nn.MaxPool2d((2, 1))
        
        #decoder for classification
        self.class_fc1=nn.Linear(self.hidden_dim*1, self.hidden_dim)
        self.class_fc2=nn.Linear(self.hidden_dim, self.hidden_dim)
        self.class_fc3=nn.Linear(self.hidden_dim, 5)
        
        #decoder for count regression
        self.count_fc3=nn.Linear(self.hidden_dim*1, 1)
        
        self.p=p
        self.drop_out=nn.Dropout(p=self.p)
        self.activation=nn.ReLU()

    def forward(self,x):
        x=x.permute(0,3,1,2)                
        
        x=self.activation(self.cnn_1(x))
        
        #encoder layer
        x=self.resi1(x)
        x=self.pool1(x)
        x=self.resi2(x)
        x=self.pool2(x)        
        x=self.resi3(x)
        x=self.pool3(x)
        x=self.resi4(x)
        x=self.pool4(x)
        x=self.resi5(x)
        x=self.pool5(x)
        x=self.resi6(x)
        x=self.pool6(x)
        
        x=self.resi7(x)
        x=self.pool7(x)
        x=self.resi8(x)
        x=self.pool8(x)  
        
        #result flatten
        x=x.reshape(-1,self.hidden_dim*1)
        
        #class classification
        class_x=self.drop_out(self.activation(self.class_fc1(x)))
        class_x=self.drop_out(self.activation(self.class_fc2(class_x)))
        logit=self.class_fc3(class_x)
        
        #count regression
        count_x = self.count_fc3(x)

        return count_x.squeeze(-1),logit