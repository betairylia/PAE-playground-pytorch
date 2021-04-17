import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

from torch_cluster import knn_graph

class kNNLayer(nn.Module):

    def __init__(self, k = 16):

        super(kNNLayer, self).__init__()

        self.k = k
    
    '''
    batch: torch_geometric.batch without edges
    returns: torch_geometric.batch with edges
    '''
    def forward(self, batch):

        edge_index = knn_graph(batch.pos, k=self.k, batch=batch.batch, loop=False)
        batch.edge_index = edge_index

        return batch
