import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .VAT import VAT_sim_KL_unnormalized

class VMTLoss(nn.Module):
    
    def __init__(
        self,
        net_getter,
        x1_getter, y1_getter,
        x2_getter = None, y2_getter = None,
        loss = lambda x, y: torch.mean(VAT_sim_KL_unnormalized(x, y, 1)),
        shuffle=True,
        samples=1):

        super(VMTLoss, self).__init__()
        
        self.net_getter = net_getter

        self.x1_getter = x1_getter
        self.y1_getter = y1_getter

        if x2_getter is None:

            print("VMT: Using x1,y1 as x2,y2 since x2,y2 == None")

            self.x2_getter = x1_getter
            self.y2_getter = y1_getter

        else:

            self.x2_getter = x2_getter
            self.y2_getter = y2_getter

        self.loss = loss

        self.shuffle = shuffle
        self.samples = samples

    def forward(self, batchContext):

        # Fetch vars

        net = self.net_getter(batchContext)

        x1 = self.x1_getter(batchContext)
        y1 = self.y1_getter(batchContext)

        x2 = self.x2_getter(batchContext)
        y2 = self.y2_getter(batchContext)

        # Calculation

        assert x1.shape[0] == x2.shape[0]
        bs = x1.shape[0]

        shape_narrowed = list(x1.shape)
        shape_narrowed = [shape_narrowed[0]] + [1 for i in range(len(shape_narrowed) - 1)]

        if self.shuffle:

            x2_idx = torch.randperm(len(x2), device = x2.device)
            x2 = x2[x2_idx]
            y2 = y2[x2_idx]
        
        xs = []
        ys = []

        for i in range(self.samples):

            alphas = torch.rand(bs, device = x1.device).unsqueeze(-1)

            xs.append(x1 * alphas.view(*shape_narrowed) + x2 * (1 - alphas.view(*shape_narrowed)))
            ys.append(y1 * alphas + y2 * (1 - alphas))

        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        
        feature_mixup, logits_mixup = net(xs)
        loss = self.loss(logits_mixup, ys.detach())

        return loss
