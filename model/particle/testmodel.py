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

        self.encoder = nn.Sequential(

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
            NetFiltersLayer(128, 16 - pos_ch, d, pos_ch, act = nn.Identity())

        )

        self.decoder = nn.Sequential(

            DecodingInputLayer(ratio = 64, latentC = 16 - pos_ch, featC = 64, outC = 16),
            FCAdaINLayer(inC = 16, outC = 64, featC = 64),
            FCAdaINLayer(inC = 64, outC = 64, featC = 64),
            FCAdaINLayer(inC = 64, outC = 3, featC = 64, act = nn.Identity())

        )

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
        
    def forward(self, batch):
       
        latent_batch = self.encoder(batch)
        decoded_batch = self.decoder(latent_batch)

        return latent_batch, decoded_batch
