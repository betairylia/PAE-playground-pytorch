'''
No longer useful
'''

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.functional as F

from torch_geometric.data import Data

class InputLayer(nn.Module):

    def __init__(self):
        super(InputLayer, self).__init__()
    
    # x: torch_geometric.data
    def forward(self, x):

        # ...

        return input_obj
