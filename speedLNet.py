
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as  F

import torchvision.transforms.functional as Ft

import torchvision
import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import math
import pathlib

def count_E(error_value, channels, batchsize):
    print(f"E = {math.sqrt(error_value/(batchsize*channels))}")

class speedLNet(nn.Module):


    def __init__(self):
         # calling super class constructor to use parent methods later
        super(speedLNet, self).__init__()
        # first convolution layer
        # in channels: 2
        # out channels: 9
        # kernel size: 5x5
        # object size 28x28, to flatten we do 28*28
        self.conv1 = nn.Conv2d(1, 16, 4)
        # 2nd out * number of calsses is the next input
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.pool = nn.MaxPool2d(2, 2)

        #[m1 = kernel size * kernelsize * outchannels  in last conv x batchsize )
        #[m2 = insize from linear, out size from linear]

        # an affine operation y = Wx + b
        # 6*6 from image dimension (?)32
        # in_features - size of each features sample
        # out_features = sieze of an output feature - image
        self.fc1 = nn.Linear(5 * 5*32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        
        #using cuda for acceleration
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        #   max pooling over a (2,2) window - take a maximum  valuefrom each feature (I think)
        #   relu applies activation function to each element of the matrix
        #   rectifier - activation function defined as the positive part of its argument
        #   also called ramp function. Rectifier - prostownik

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # channels * width * height
        x = x.view(-1, 1*5*5*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

  

if __name__ == "__main__":
    count_E(2304, 1, 1)



    