import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def VAT_sim_KL_unnormalized(logitA, logitB, eps):
    A_B = F.softmax(logitA, dim = -1) * (F.log_softmax(logitA, dim = -1) - F.log_softmax(logitB, dim = -1))
    loss = torch.sum(A_B, 1)
    return loss

def autoNormalize(X):
    s = X.shape
    return F.normalize(X.view(s[0], -1), p = 2, dim = 1).view(s)

# VAT helpers
# TODO: disable BN
def getVAT(forward, x, VAT_sim = VAT_sim_KL_unnormalized, xi = 10, eps = 1.0, logits = None):

    VAT_d = torch.randn_like(x, device = x.device)
    # VAT_d = torch.tensor(np.random.randn(*x.shape).astype('f'), requires_grad = True).to(x.device)
    VAT_d = autoNormalize(VAT_d)
    VAT_r = xi * VAT_d
    # VAT_r = VAT_r.to(x.device)
    VAT_r = VAT_r.requires_grad_()
    VAT_r.retain_grad()

    if logits is None:
        _, logits = forward(x)
    
    target = logits.detach()

    r_x = x + VAT_r
    _, out_vadv = forward(r_x)

    VAT_kl = torch.sum(VAT_sim(target, out_vadv, eps))
    VAT_g = torch.autograd.grad(VAT_kl, VAT_r)[0]
    VAT_r_vadv = eps * autoNormalize(VAT_g.detach())

    return VAT_r_vadv

class VATLoss(nn.Module):

    def __init__(
        self,
        net_getter, logit_getter, x_getter, VAT_eps_getter=lambda bc: None,
        xi=1.0, eps=3.5, use_variable_eps=False,
        diff_func=VAT_sim_KL_unnormalized):
        
        super(VATLoss, self).__init__()

        self.x_getter = x_getter
        self.logit_getter = logit_getter
        self.net_getter = net_getter
        self.VAT_eps_getter = VAT_eps_getter

        self.xi = xi
        self.eps = eps
        self.use_variable_eps = use_variable_eps

        self.diff_func = diff_func

        print("VAT initialized with xi = %f, eps = %s" % (self.xi, ("%f" % self.eps) if use_variable_eps == False else "Variable"))

    def forward(self, batchContext):

        # Fetch vars

        x = self.x_getter(batchContext)
        net = self.net_getter(batchContext)
        logits = self.logit_getter(batchContext)
        adapt_VAT = self.VAT_eps_getter(batchContext)

        # VAT calculation

        if self.use_variable_eps:
            std = torch.ones(x.shape[0], 1).to(x.device) * (adapt_VAT).unsqueeze(-1)
        else:
            std = self.eps

        r = getVAT(net, x, self.diff_func, self.xi, std, logits = logits)
        _, logits_p = net(x + r)

        loss = self.diff_func(logits.detach(), logits_p, std)

        return torch.mean(loss)

def ShannonEntropy(x):
    return torch.mean(torch.sum(-x * torch.log(x + 1e-10), -1))