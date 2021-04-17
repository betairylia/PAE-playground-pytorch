import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math

# Input = BS x 3 x 224 x 224
class ResNet_wrapper(nn.Module):

    def __init__(self, bn = True, IN = True):
    
        super(ResNet_wrapper, self).__init__()
        
        self.embedder = models.resnet18(pretrained=True)
        self.embedder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.embedder.fc = nn.Linear(512, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 10),
            # nn.Softmax(dim=-1),
        )

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

        x = self.embedder(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features