import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as TF

import math
from .SM_Layers import Linear_SM, Conv2d_SM

class CONV_2(nn.Module):
    def __init__(self):
        super(CONV_2, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=3,padding=1,out_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64,kernel_size=3,padding=1,out_channels=64)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=16384,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=256)
        self.fc3 = nn.Linear(in_features=256,out_features=10)

        self.dropout = nn.Dropout()

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 16384))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

class CONV_4(nn.Module):
    def __init__(self):
        super(CONV_4, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=3,padding=1,out_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64,kernel_size=3,padding=1,out_channels=64)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64,kernel_size=3,padding=1,out_channels=128)
        self.conv4 = nn.Conv2d(in_channels=128,kernel_size=3,padding=1,out_channels=128)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=8192,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=256)
        self.fc3 = nn.Linear(in_features=256,out_features=10)

        self.dropout = nn.Dropout()

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = TF.relu(x)

        x = self.conv4(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 8192))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

class CONV_6(nn.Module):
    def __init__(self):
        super(CONV_6, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=3,padding=1,out_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64,kernel_size=3,padding=1,out_channels=64)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64,kernel_size=3,padding=1,out_channels=128)
        self.conv4 = nn.Conv2d(in_channels=128,kernel_size=3,padding=1,out_channels=128)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(in_channels=128,kernel_size=3,padding=1,out_channels=256)
        self.conv6 = nn.Conv2d(in_channels=256,kernel_size=3,padding=1,out_channels=256)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=4096,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=256)
        self.fc3 = nn.Linear(in_features=256,out_features=10)

        self.dropout = nn.Dropout()

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = TF.relu(x)

        x = self.conv4(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv5(x)
        x = TF.relu(x)

        x = self.conv6(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 4096))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

class CONV_2_SM(nn.Module):
    def __init__(self, K, score_selection):
        super(CONV_2_SM, self).__init__()
        

        self.conv1 = Conv2d_SM(in_channels=3,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)
        self.conv2 = Conv2d_SM(in_channels=64,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = Linear_SM(in_features=16384,out_features=256, K=K, score_selection=score_selection)
        self.fc2 = Linear_SM(in_features=256,out_features=256, K=K, score_selection=score_selection)
        self.fc3 = Linear_SM(in_features=256,out_features=10, K=K, score_selection=score_selection)

        self.dropout = nn.Dropout()

    def update_scores(self,lr):
        self.conv1.update_scores(lr)
        self.conv2.update_scores(lr)
        self.fc1.update_scores(lr)
        self.fc2.update_scores(lr)
        self.fc3.update_scores(lr)

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 16384))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

class CONV_4_SM(nn.Module):
    def __init__(self, K, score_selection):
        super(CONV_4_SM, self).__init__()
        

        self.conv1 = Conv2d_SM(in_channels=3,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)
        self.conv2 = Conv2d_SM(in_channels=64,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = Conv2d_SM(in_channels=64,kernel_size=3,padding=1,out_channels=128, K=K, score_selection=score_selection)
        self.conv4 = Conv2d_SM(in_channels=128,kernel_size=3,padding=1,out_channels=128, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = Linear_SM(in_features=8192,out_features=256, K=K, score_selection=score_selection)
        self.fc2 = Linear_SM(in_features=256,out_features=256, K=K, score_selection=score_selection)
        self.fc3 = Linear_SM(in_features=256,out_features=10, K=K, score_selection=score_selection)

        self.dropout = nn.Dropout()

    def update_scores(self,lr):
        self.conv1.update_scores(lr)
        self.conv2.update_scores(lr)
        self.conv3.update_scores(lr)
        self.conv4.update_scores(lr)
        self.fc1.update_scores(lr)
        self.fc2.update_scores(lr)
        self.fc3.update_scores(lr)

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = TF.relu(x)

        x = self.conv4(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 8192))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

class CONV_6_SM(nn.Module):
    def __init__(self, K, score_selection):
        super(CONV_6_SM, self).__init__()
        

        self.conv1 = Conv2d_SM(in_channels=3,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)
        self.conv2 = Conv2d_SM(in_channels=64,kernel_size=3,padding=1,out_channels=64, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = Conv2d_SM(in_channels=64,kernel_size=3,padding=1,out_channels=128, K=K, score_selection=score_selection)
        self.conv4 = Conv2d_SM(in_channels=128,kernel_size=3,padding=1,out_channels=128, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = Conv2d_SM(in_channels=128,kernel_size=3,padding=1,out_channels=256, K=K, score_selection=score_selection)
        self.conv6 = Conv2d_SM(in_channels=256,kernel_size=3,padding=1,out_channels=256, K=K, score_selection=score_selection)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = Linear_SM(in_features=4096,out_features=256, K=K, score_selection=score_selection)
        self.fc2 = Linear_SM(in_features=256,out_features=256, K=K, score_selection=score_selection)
        self.fc3 = Linear_SM(in_features=256,out_features=10, K=K, score_selection=score_selection)

        self.dropout = nn.Dropout()

    def update_scores(self,lr):
        self.conv1.update_scores(lr)
        self.conv2.update_scores(lr)
        self.conv3.update_scores(lr)
        self.conv4.update_scores(lr)
        self.conv5.update_scores(lr)
        self.conv6.update_scores(lr)
        self.fc1.update_scores(lr)
        self.fc2.update_scores(lr)
        self.fc3.update_scores(lr)

    def forward(self,x):

        bs = x.shape[0]

        x = self.conv1(x)
        x = TF.relu(x)

        x = self.conv2(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = TF.relu(x)

        x = self.conv4(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = self.conv5(x)
        x = TF.relu(x)

        x = self.conv6(x)
        x = TF.relu(x)

        x = self.maxpool(x)

        x = torch.reshape(x,(bs, 4096))

        x = self.fc1(x)
        x = TF.relu(x)
        x = self.dropout(x)
        

        x = self.fc2(x)
        x = TF.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x