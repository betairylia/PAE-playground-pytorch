import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random

import torch_geometric.data

import os
import numpy as np

class ParticlesDataset(Dataset):

    '''
    Datasets for load particle data.
    Ideally, raw point cloud data should be stored in hdf5, with their (first) k-NNG precomputed.
    For now, k-NNG is computed on the fly, after batch sent to GPU.
    '''

    def __init__(self, positions, features = None, labels = None, n_classes = -1):
        
        self.positions = positions
        self.features = features
        self.n_classes = n_classes
        self.labels = labels

        self.size = self.positions.shape[0]
        
    def __getitem__(self, idx):

        idx = idx if idx < self.size else random.randint(0, self.size - 1)

        pos = torch.FloatTensor(self.positions[idx])

        # use fake features if features is empty
        if self.features is not None:
            fea = torch.FloatTensor(self.features[idx])
        else:
            fea = torch.zeros(pos.shape[0], 1)

        data = torch_geometric.data.Data(
            x=fea,
            pos=pos
        )

        return data
    
    def __len__(self):
        return self.size
