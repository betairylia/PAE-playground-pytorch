import torch
from torch import nn

class VanillaAdversarialLoss(nn.Module):

    def __init__(self, dscore_getter, isFake = False):

        super(VanillaAdversarialLoss, self).__init__()
        self.dscore_getter = dscore_getter
        self.isFake = isFake

    def forward(self, batchContext):

        score = self.dscore_getter(batchContext)

        if self.isFake == True:
            ref = torch.zeros_like(score)
        else:
            ref = torch.ones_like(score)

        loss = torch.nn.functional.binary_cross_entropy(score, ref.float())
        return loss

class WassersteinAdversarialLoss(nn.Module):

    def __init__(self, dscore_getter, isFake = False):

        super(WassersteinAdversarialLoss, self).__init__()
        self.dscore_getter = dscore_getter
        self.isFake = isFake

    def forward(self, batchContext):

        score = self.dscore_getter(batchContext)

        if self.isFake == True:
            loss = torch.mean(score)
        else:
            loss = - torch.mean(score)

        return loss

class GradientPenalty(nn.Module):

    def __init__(self, discriminator_getter, x1_getter, x2_getter, GP_samples = 1):

        super(GradientPenalty, self).__init__()

        self.discriminator_getter = discriminator_getter

        self.x1_getter = x1_getter
        self.x2_getter = x2_getter

        self.GP_samples = GP_samples

    def forward(self, batchContext):

        x1 = self.x1_getter(batchContext)
        x2 = self.x2_getter(batchContext)
        d = self.discriminator_getter(batchContext)

        assert x1.shape[0] == x2.shape[0]
        bs = x1.shape[0]

        # pts = [x1, x2]
        pts = []
        
        for i in range(self.GP_samples):
            gp_alpha = torch.rand(bs, 1, device = x1.device)
            gp_pts = x1.detach() * gp_alpha + x2.detach() * (1 - gp_alpha)
            pts.append(gp_pts)

        pts = torch.cat(pts, 0).detach()
        gp_pts = torch.autograd.Variable(pts, requires_grad=True)
        
        gp_outs = d(gp_pts)
        gradients = torch.autograd.grad(
            gp_outs, gp_pts,
            grad_outputs=torch.ones(gp_outs.size(), device = gp_outs.device),
            create_graph = True)[0]
        gradients = gradients.view(bs, -1)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim = 1) + 1e-12)
                
        gp_loss = ((gradients_norm - 1) ** 2).mean()
        
        return gp_loss
