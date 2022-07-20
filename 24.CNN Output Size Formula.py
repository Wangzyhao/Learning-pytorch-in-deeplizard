from numpy import identity
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self,t):
        #(1)input layer
        t = t   
        
        #(2)hidden conv layer
        t = self.conv1(t)                       
        t = F.relu(t)                           
        t = F.max_pool2d(t, kernel_size=2, stride=2)   #只会使高度和宽度的维度下降
        
        #(3)hidden conv layer
        t = self.conv2(t)                             
        t = F.relu(t)                           
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #(4)hidden linear layer
        t = t.reshape(-1,12*4*4)       #-1表示系统自动计算对应的值
        t = self.fc1(t)                
        t = F.relu(t)
        
        #(5)hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        
        #(6)output layer
        t = self.out(t)
        #t = F.softmax(t,dim=1)             
                                            
        return t
        
        

network = Network()

train_set = torchvision.datasets.FashionMNIST(
    root = 'data/FashionMNIST',
    train = True,
    download =True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)


sample = next(iter(train_set))
image,label = sample

output = network(image.unsqueeze(0))
print(output)




### CNN Output Size Formlua
### O = [(n-f+2p)/s]+1

###Operation                   #Output Shape
                 
#Identity function           torch.Size([1,1,28,28])

#Convolution (5*5)           torch.Size([1,6,24,24])     [(28-5+2*0)/1]+1

#Maxpooling  (2*2)           torch.Size([1,6,12,12])     [(24-2+2*0)/2]+1

#Convolution (5*5)           torch.Size([1,12,8,8])      [(12-5+2*0)/1]+1

#Maxpooling  (2*2)           torch.Size([1,12,4,4])

#Flatten(reshape)            torch.Size([1,192])

#Linear transformation       torch.Size([1,120])

#Linear transformation       torch.Size([1,60])

#Linear transformation       torch.Size([1,10])

