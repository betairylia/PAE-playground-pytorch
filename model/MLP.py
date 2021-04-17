import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

class PlainMLP(nn.Module): 

    def __init__(self, dim, C):
        super(PlainMLP, self).__init__()

        self.__in_features = 256

        self.l1 = nn.Linear(dim, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.l2 = nn.Linear(1200,self.__in_features)
        self.bn2 = nn.BatchNorm1d(self.__in_features)
        self.l3 = nn.Linear(self.__in_features,C)
        self.bn3 = nn.BatchNorm1d(C)

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
        
    def forward(self, x):
       
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        # x = self.bn3(self.l3(x))
        y = self.l3(x)

        return x, y

    def output_num(self):
        return self.__in_features