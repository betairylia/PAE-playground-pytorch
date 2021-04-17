import torch
from torch import nn

import wandb

class BinaryLossOp(nn.Module):

    def __init__(
        self,
        left_getter = lambda bc: bc.source.prediction,
        right_getter = lambda bc : bc.source.y,
        loss_func = torch.nn.CrossEntropyLoss()):
        
        super(BinaryLossOp, self).__init__()

        self.loss_func = loss_func
        self.left_getter = left_getter
        self.right_getter = right_getter

    def forward(self, batchContext):
        loss = self.loss_func(self.left_getter(batchContext), self.right_getter(batchContext))
        return loss

class LossComposition(nn.Module):

    def __init__(self, losses):

        '''
        losses should be an array of objects:
        [
            {'name': "SomeLoss1", 'loss': LossA(), 'weight': opt.lambda1},
            {'name': "SomeLoss2", 'loss': LossB(), 'weight': opt.lambda2},
            ...
        ]
        '''

        super(LossComposition, self).__init__()

        self.losses = losses

        for loss in self.losses:

            if 'name' not in loss:
                loss['name'] = "Noname"

            if 'weight' not in loss:
                loss['weight'] = 1.0

            loss['val'] = 0
            loss['totalValue'] = 0

        self.iters = 0
    
    def Add(self, loss):
        self.losses.append(loss)

    def forward(self, batchContext):

        total_loss = 0

        for loss in self.losses:

            if loss['weight'] > 0:

                loss['value'] = loss['loss'](batchContext) * loss['weight']
                loss['totalValue'] += loss['value'].detach().cpu()
            
                total_loss += loss['value']

        self.iters += 1

        return total_loss

    def LogToWandB(self, myName = 'loss', resetAverages = False):

        obj = {}
        obj["%s/all" % (myName)] = 0

        for loss in self.losses:
            
            obj["%s/%s" % (myName, loss['name'])] = (loss['totalValue'] / self.iters) if self.iters > 0 else 0
            obj["%s/all" % (myName)] += obj["%s/%s" % (myName, loss['name'])]

            if resetAverages == True:
                loss['totalValue'] = 0

        if resetAverages == True:
            self.iters = 0

        wandb.log(obj)
        return

    def GetLogString(self, resetAverages = True):

        result = ""
        for loss in self.losses:
            result += "%s: %+8.4f | " % (loss['name'], (loss['totalValue'] / self.iters) if self.iters > 0 else 0)
            if resetAverages == True:
                loss['totalValue'] = 0

        result = result[:-3]

        if resetAverages == True:
            self.iters = 0

        return result
