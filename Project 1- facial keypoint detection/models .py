## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## input images are 224x224 pixels
        # (W-F)/S=(224-5)1+1=220, pooling 110x110, 32 channels
        self.conv1 = nn.Conv2d(1, 32, 5)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # output tensor (32, 110, 110)
        self.fc_drop1 = nn.Dropout(p=0.2)
        
        
        
        # (110-5)/1+1=106
        self.conv2 = nn.Conv2d(32, 36, 5)
        # output (24, 106,106)
        # max pulling: (24,53,53)
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_drop2 = nn.Dropout(p=0.2)
        
                
        # (53-5)/1+1=49
        self.conv3 = nn.Conv2d(36, 48, 5)
        # output (48, 49,49)
        # (48,24,24) 
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc_drop3 = nn.Dropout(p=0.2)
        
        # (24-3)/1+1=22
        self.conv4 = nn.Conv2d(48, 64, 3)
        # output (64, 22,22)
        # max pulling: (64,11,11)
        # pool with kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc_drop4 = nn.Dropout(p=0.2)
        
        
        # (11-3)/1+1=9
        self.conv5 = nn.Conv2d(64, 64, 3)
        # output (64, 9,9)
        # max pulling: (64,4,4)
        # pool with kernel_size=2, stride=2
        self.pool5 = nn.MaxPool2d(2, 2)
   
     
        
        
        
              
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
           # dropout with p=0.4
        
        
        #self.fc1_drop2 = nn.Dropout(p=0.3)
        self.fc6 = nn.Linear(64*4*4, 136)
       
        ###self.fc7 = nn.Linear(1360, 136)
        
        # dropout with p=0.4
        ## self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        ## self.fc2 = nn.Linear(680, 136)
        
        ## self.fc3 = nn.Linear(680, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.fc_drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc_drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc_drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.fc_drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        
        x = x.view(x.size(0), -1)
        # two linear layers with dropout in between
        ##x = F.relu(self.fc1(x))
        #x = self.fc1_drop2(x)
        x = self.fc6(x)
        ###x = self.fc7(x)
        
        
     
        
        # final output
        
        
        ## TODO: define the convolutional neural network architecture


        # a modified x, having gone through all the layers of your model, should be returned
        return x
        