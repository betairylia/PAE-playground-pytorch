import torch
import torch.nn as nn
import numpy as np
from .DiscriminatorBackBone import Discriminator, SNDiscriminator

'''
TODO: Complete this; current obstacle := cannot obtain datapoints used for GP since we don't have same number of "real" and "fake" (we even cannot access them in seperated version easily)
Current solution - random sample same amount of points from both minibatches for GP calculation.
'''
class WGANdis(nn.Module):

    def __init__(self, bnD, in_feature, hidden_size, GP_lambda = 10.0, GP_samples = 1):

        super(WGANdis, self).__init__()
        self.net = SNDiscriminator(in_feature, hidden_size)

    def forward(self, x):

        score = self.net(x).squeeze(-1)

        return score
