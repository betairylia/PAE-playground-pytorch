import torch
import torch.nn as nn
import numpy as np
from .specnorm import SNLinear

# TODO: Really need this ?
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        # nn.init.zeros_(m.bias)
        m.bias.data.fill_(0)

def block(in_c, out_c, linear = nn.Linear, norm = None, act = nn.ReLU()):
    layers=[
        linear(in_c,out_c),
        norm(out_c) if norm is not None else nn.Identity(),
        act if act is not None else nn.Identity()
    ]
    return layers

class GeneralMLP(nn.Module):

    def __init__(self, layers, act=nn.ReLU(), linear=nn.Linear, norm=None):
        
        super(GeneralMLP, self).__init__()

        net = []
        for i in range(len(layers) - 2):
            net += block(layers[i], layers[i + 1], linear, norm, act)
        
        net += block(layers[-2], layers[-1], linear, norm, None)

        self.net = nn.Sequential(*net)

    def forward(self, x):

        return self.net(x)

class Discriminator(nn.Module):

    def __init__(self, in_feature, hidden_size):

        super(Discriminator, self).__init__()
        self.net = GeneralMLP([in_feature, hidden_size, 1])

    def forward(self, x):

        y = self.net(x)
        return y

    def output_num(self):
        return 1

    # def get_parameters(self):
    #     return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]

class SNDiscriminator(nn.Module):

    def __init__(self, in_feature, hidden_size):

        super(SNDiscriminator, self).__init__()
        self.ad_layer1 = SNLinear(in_feature, hidden_size)
        self.ad_layer2 = SNLinear(hidden_size, hidden_size)
        self.ad_layer3 = SNLinear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.apply(init_weights)

    def forward(self, x):

        x = x * 1.0
        x = self.ad_layer1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        # x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]