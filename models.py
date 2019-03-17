## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
# This model is based on NamishNet (https://arxiv.org/pdf/1710.00977.pdf)
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #conv layers
        ## output size = (W-F)/S +1 (see output sizes specified in the forward function)
        # (1, 32, kernel_size=3x3, stride=(1, 1))
        self.conv1 = nn.Conv2d(  1, 32, 3)       
        # (32, 64, kernel_size=3x3, stride=(1, 1))
        self.conv2 = nn.Conv2d( 32, 64, 3)        
        # (64, 128, kernel_size=2x2, stride=(1, 1))
        self.conv3 = nn.Conv2d( 64,128, 2)  
        # (128, 256, kernel_size=2x2, stride=(1, 1))
        self.conv4 = nn.Conv2d(128,256, 2)        
        
        # maxpool layer with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)            
        
        # dropout layers with varying levels of dropouts
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.3)
        self.drop3 = nn.Dropout(p=0.4)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.5)
        
        # linear
        # input channels = 256 * 12 * 12 = 36864, output channels = 1000
        self.fc1 = nn.Linear(36864, 1000)
        # input channels = 1000, output channels = 1000
        self.fc2 = nn.Linear(1000, 1000)
        # input channels = 1000, output channels = 136
        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):

        # Size - (1,224,224)
        # conv layers (with maxpooling and dropouts)
        # After Conv - W = (224-3)/1 +1 = 222. Output Size = (32,222,222)
        # After maxpool (and dropout) 222/2 = 111. Output Size = (32,111,111)
        x = self.drop1(self.pool(F.relu(self.conv1(x))))  
        # After Conv - (111-3)/1 +1 = 109. Output Size = (64,109,109)
        # After maxpool (and dropout) 109/2 = 54. Output Size = (64,54,54)
        x = self.drop2(self.pool(F.relu(self.conv2(x))))  
        # After Conv - (54-2)/1 +1 = 53. Output Size = (128,53,53)
        # After maxpool (and dropout) 53/2 = 26. Output Size = (128,26,26)
        x = self.drop3(self.pool(F.relu(self.conv3(x))))  
        # After Conv - (26-2)/1 +1 = 25. Output Size = (256,25,25)
        # After maxpool (and dropout) 25/2 = 12(rounded down). Output Size = (256,12,12) 
        x = self.drop4(self.pool(F.relu(self.conv4(x))))  

        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # linear layers (with dropouts)
        # input channels = 256 * 12 * 12 = 36864, output channels = 1000
        x = self.drop5(F.relu(self.fc1(x)))
        # input channels = 1000, output channels = 1000
        x = self.drop6(F.relu(self.fc2(x)))
        # input channels = 1000, output channels = 136
        x = self.fc3(x)
        
        # final output
        return x
