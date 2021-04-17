'''TODO: Check this, just copied to here'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SNLinear(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True, init_u=None, use_gamma=False
    ):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.Ip = 1
        self.register_buffer(
            "u", init_u if init_u is not None else torch.randn(1, out_features)
        )
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None

        self.Ip_grad = 8
        self.r = 10
        self.register_buffer(
            "u0", init_u if init_u is not None else torch.randn(1, out_features)
        )
        self.register_buffer(
            "u1", init_u if init_u is not None else torch.randn(1, out_features)
        )

    @property
    def W_bar(self):
        sigma, u, _ = max_singular_value(self.weight, self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def forward(self, x):
        if self.gamma is not None:
            return torch.exp(self.gamma) * F.linear(x, self.W_bar, self.bias)
        else:
            return F.linear(x, self.W_bar, self.bias)

def max_singular_value(weight, u, Ip):
    assert Ip >= 1

    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1).detach()
        _u = F.normalize(torch.mm(_v, weight.transpose(0, 1)), p=2, dim=1).detach()
    sigma = torch.sum(F.linear(_u, weight.transpose(0, 1)) * _v)
    return sigma, _u, _v