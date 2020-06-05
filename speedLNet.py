
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


class speedLNet(nn.Module):


    def __init__(self):
         # calling super class constructor to use parent methods later
        super(speedLNet, self).__init__()
        # first convolution layer
        # in channels: 2
        # out channels: 9
        # kernel size: 5x5
        
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # an affine operation y = Wx + b
        # 6*6 from image dimension (?)
        # in_features - size of each features sample
        # out_features = sieze of an output feature - image
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
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
        x = x.view(-1, 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    


    