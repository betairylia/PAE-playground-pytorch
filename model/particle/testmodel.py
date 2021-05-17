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

        d = 2
        pos_ch = 3

        self.net = nn.Sequential(

            kNNLayer(k=16),

            NetFiltersLayer(1, 32, d, pos_ch),
            NetFiltersLayer(32, 32, d, pos_ch),
            NetFiltersLayer(32, 32, d, pos_ch),

            FarthestSamplingLayer(ratio=0.25),
            kNNLayer(k=16),

            NetFiltersLayer(32, 64, d, pos_ch),
            NetFiltersLayer(64, 64, d, pos_ch),
            NetFiltersLayer(64, 64, d, pos_ch),
            NetFiltersLayer(64, 64, d, pos_ch),

            FarthestSamplingLayer(ratio=0.0625),
            kNNLayer(k=16),

            NetFiltersLayer(64, 128, d, pos_ch),
            NetFiltersLayer(128, 128, d, pos_ch),
            NetFiltersLayer(128, 128, d, pos_ch),
            NetFiltersLayer(128, 128, d, pos_ch),
            NetFiltersLayer(128, 128, d, pos_ch),
            NetFiltersLayer(128, 128, d, pos_ch),
            NetFiltersLayer(128, 16 - pos_ch, d, pos_ch),
        )

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
        
    def forward(self, batch):
       
        batch = self.net(batch)

        return batch
