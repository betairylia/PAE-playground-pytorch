import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

from torch_cluster import fps
import torch_geometric.data

from torch_scatter import scatter

class DecodingInputLayer(nn.Module):

    def __init__(self, ratio = 64, latentC = 13, featC = 32, outC = 8):

        super(DecodingInputLayer, self).__init__()

        self.ratio = ratio
        self.featNet = nn.Linear(latentC, featC)
        self.outC = outC

    def duplicate(self, tensor, N):

        o_shape = list(tensor.shape)
        new_shape = [o_shape[0]] + [1] + o_shape[1:]
        repeat = [1] + [N] + [1 for i in range(tensor.dim() - 1)]

        o_shape[0] *= N

        return tensor.view(new_shape).repeat(*repeat).view(o_shape)

    '''
    batch: torch_geometric.batch without edges
    returns: torch_geometric.batch with edges
    '''
    def forward(self, batch):

        # Copy Tensors N Times
        latent_point_batch = self.duplicate(torch.arange(batch.batch.shape[0], device = batch.x.device), self.ratio)
        origin_pos = batch.pos

        # Handle features
        feat = F.relu(self.featNet(batch.x))
        origin_feat = feat
        
        # Generate Noise
        noise = torch.randn(latent_point_batch.shape[0], self.outC, device = batch.x.device)

        # Create batch
        new_batch = torch_geometric.data.Batch(
            batch = batch.batch,                            # per Latent-point
            latent_point_batch = latent_point_batch,        # per Point
            x = noise,                                      # per Point
            origin_feat = origin_feat,                      # per Latent-point
            origin_pos = origin_pos                         # per Latent-point
        )

        return new_batch

class FCAdaINLayer(nn.Module):

    def __init__(self, inC = 16, outC = 3, featC = 32, act = nn.ReLU()):

        super(FCAdaINLayer, self).__init__()

        self.fc = nn.Linear(inC, outC)
        self.feature_mu = nn.Linear(featC, outC)
        self.feature_sig = nn.Linear(featC, outC)

        self.act = act

    def forward(self, batch):

        x = self.fc(batch.x)

        mu = self.feature_mu(batch.origin_feat)
        sig = self.feature_sig(batch.origin_feat)

        # AdaIN
        x_mean = scatter(x, batch.latent_point_batch.unsqueeze(-1), reduce = "mean", dim = 0)
        x_normalized = x - x_mean[batch.latent_point_batch]

        x_std = scatter(
            (x_normalized) ** 2, 
            batch.latent_point_batch.unsqueeze(-1), 
            reduce = "mean",
            dim = 0
        )
        x_normalized = x_normalized / torch.sqrt(x_std[batch.latent_point_batch] + 1e-14)

        x = x_normalized * sig[batch.latent_point_batch] + mu[batch.latent_point_batch]
        
        # Act
        x = self.act(x)

        batch.x = x

        return batch

# class OutputAssemblyLayer(nn.Module):

#     def __init__(self):

#         super(OutputAssemblyLayer, self).__init__()
    
#     def forward(self, batch):


