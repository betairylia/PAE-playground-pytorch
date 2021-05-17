from .PyTorchOT import *

class SinkhornLoss(nn.Module):

    def __init__(
        self,

        # TODO: below getters cannot get [Batchsize, N, C] point cloud tensors;
        # instead they get a concatenated [N*bs, C] torch_geometry stuff ...
        left_getter = lambda bc: bc.x.pos,
        right_getter=lambda bc: bc.reconstruction.pos,
        
        eps=1e-2,
        max_iter=100):
        
        super(SinkhornLoss, self).__init__()

        self.left_getter = left_getter
        self.right_getter = right_getter

        self.loss_func = SinkhornDistance(eps = eps, max_iter = max_iter, reduction = 'mean')

    def forward(self, batchContext):

        # Fetch data
        ref = self.left_getter(batchContext)
        rec = self.right_getter(batchContext)

        # Calculation
        loss, _, _ = self.loss_func(ref, rec)

        return loss