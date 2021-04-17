import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math

# For SVHN dataset
# Input = BS x 3 x 32 x 32
class DTN(nn.Module):
    
    def __init__(self, bn=True, IN=True):
        
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

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
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features