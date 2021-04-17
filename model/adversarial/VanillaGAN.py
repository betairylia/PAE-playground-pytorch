import torch
import torch.nn as nn
import numpy as np
from .DiscriminatorBackBone import Discriminator

class VanillaGANdis(nn.Module):

    def __init__(self, in_feature, hidden_size):

        super(VanillaGANdis, self).__init__()
        self.net = Discriminator(in_feature, hidden_size)

    def forward(self, x):

        score = self.net(x).squeeze(-1)
        score = torch.sigmoid(score)

        return score