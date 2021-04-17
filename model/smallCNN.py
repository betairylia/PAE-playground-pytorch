import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

# French et al. for CIFAR10 <-> STL
# Input = BS x 3 x 32 x 32

class SmallGAPConv(nn.Module):

    def __init__(self, bn = True, IN = True, nOut = 10):
        super(SmallGAPConv, self).__init__()
        self.conv_params = nn.Sequential(
            
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2), # 16 x 16 x 128
            nn.Dropout2d(0.5),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2), # 8 x 8 x 256
            nn.Dropout2d(0.5),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512) if bn else nn.Identity(),
            nn.ReLU(), # 6 x 6 x 512
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(), # 6 x 6 x 256
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ReLU(), # 6 x 6 x 128
        )

        # self.fc_params = nn.Linear(128, 10)

        self.classifier = nn.Linear(128, nOut)
        self.__in_features = 128

        self.IN = IN

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x):
        
        if self.IN:
            original_shape = x.shape
            x = x.view(x.shape[0], x.shape[1], -1)
            x = (x - torch.mean(x, dim = -1, keepdim = True)) / torch.std(x, dim = -1, keepdim = True)
            x = x.view(original_shape)

        x = self.conv_params(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, -1) # Global average pooling
        # x = x.view(x.size(0), -1)
        # x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features