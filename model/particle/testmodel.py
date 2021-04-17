import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

from .layers import *

class TestModel(nn.Module): 

    def __init__(self):

        super(TestModel, self).__init__()

        self.net = nn.Sequential(
            kNNLayer(k=16),
            FarthestSamplingLayer(ratio=0.5),
        )

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
        
    def forward(self, batch):
       
        batch = self.net(batch)

        return batch
