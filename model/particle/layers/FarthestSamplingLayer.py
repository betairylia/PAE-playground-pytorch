import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

from torch_cluster import fps
import torch_geometric.data

class FarthestSamplingLayer(nn.Module):

    def __init__(self, ratio = 0.5):

        super(FarthestSamplingLayer, self).__init__()

        self.ratio = ratio
    
    '''
    batch: torch_geometric.batch without edges
    returns: torch_geometric.batch with edges
    '''
    def forward(self, batch):

        indices = fps(batch.pos, batch.batch, ratio=self.ratio, random_start=False)
        
        new_batch = torch_geometric.data.Batch(
            batch=batch.batch[indices],
            x=batch.x[indices],
            pos=batch.pos[indices]
        )

        return new_batch
