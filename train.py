# TODO: Default model selection, check model competible with dataset, etc.
'''
Boilerpolate script for torpaz training
v 0.1
'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import os
import loss as loss_func
import numpy as np
import random
from datetime import datetime

from dataio import *
from model import *
from utils import *
from loss import *

from torch_geometric.data import DataLoader
from loss.PyTorchOT import SinkhornDistance

import sys
import time

import types

import wandb

def OptimStep(batchContext, lossFunc, optim):

    if not isinstance(optim, list):
        optim = [optim] 

    for o in optim:
        o.zero_grad()
    
    loss = lossFunc(batchContext)
    loss.backward()

    for o in optim:
        o.step()

def main():

    ##########################################################################
    ''' Argparse '''
    ##########################################################################

    parser = argparse.ArgumentParser(description='Torpaz boilerpolate')

    # Set-up
    parser.add_argument('--name', type=str, default='Untitled')
    parser.add_argument('--proj', type=str, default='PAE')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='test', choices=['test'], help='Model to use')

    # Training (General HPs)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='Learning Rate')

    # Misc
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nworkers', type=int, default=8, help='Dataloader num_workers.')

    # Worker meta-data
    parser.add_argument('--runid', type=int, default=0, help='Run ID.')
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')
    parser.add_argument('--wandb', type=int, default=1, help='Use WandB.')

    # TODO: Fixed seed ? Randomness ?

    args = parser.parse_args()

    # Log into WandB
    if args.wandb:
        wandb.init(project = args.proj, config = vars(args), group = GetArgsStr(args))

    ##########################################################################
    ''' Dataset '''
    ##########################################################################

    # Toy data
    positions = np.random.normal(loc = 0.0, scale = 1.0, size = (100, 4096, 3))

    # DataLoader for training
    train_loader = DataLoader(
        ParticlesDataset(positions),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nworkers
    )

    # New toy data
    positions = np.random.normal(loc=0.0, scale=1.0, size=(100, 4096, 3))
    
    # DataLoader for testing ( /validation )
    test_loader = DataLoader(
        ParticlesDataset(positions),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.nworkers
    )

    ##########################################################################
    ''' Model '''
    ##########################################################################

    net = GetParticleModel(
        modelname=args.model
    )

    ##########################################################################
    ''' Contexts initialization '''
    ##########################################################################

    # Batch Context
    bc = types.SimpleNamespace()
    bc.net = net

    # Epoch Context
    ec = types.SimpleNamespace()
    ec.net = net
    ec.train_loader = train_loader
    ec.test_loader = test_loader

    ##########################################################################
    ''' Optimizers '''
    ##########################################################################

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    optimizers = [optimizer]

    # TODO: LR scheduler

    ##########################################################################
    ''' Loss Composition '''
    ##########################################################################

    # point cloud size
    N = 4096

    # TODO: proper loss for particles
    loss = LossComposition([
        {
            # Sinkhorn reconstruction loss
            'name': "Sinkhorn",
            'weight': 1.0,
            'loss': SinkhornLoss(
                left_getter = lambda bc : bc.x.pos.view(-1, 4096),
                right_getter = lambda bc : bc.decoded.x.view(-1, 4096)
            )
        },
    ])

    # Test loss func
    # loss = LossComposition([
    #     {
    #         # Sinkhorn reconstruction loss
    #         'name': "Test",
    #         'weight': 1.0,
    #         'loss': lambda bc : torch.abs(bc.latent.x.mean() - 0.0)
    #     },
    # ])

    ##########################################################################
    ''' Main Loop '''
    ##########################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    it = 0

    for epoch in range(1, args.epochs + 1):

        # TODO: Change this to lr scheduler

        '''TRAINING'''
        net.train()
        bit = 0

        for batch in train_loader:

            # Fetch data
            batch = batch.to(device)
            bc.x = batch

            # Regular update
            # breakpoint()
            bc.latent, bc.decoded = bc.net(bc.x)

            # Test
            # pred = bc.decoded.x[:4096].unsqueeze(0)
            # true = batch.pos[:4096].unsqueeze(0)
            # sink = SinkhornDistance(1e-2, 100)
            # print(sink(pred, true))

            # breakpoint()
            
            OptimStep(bc, loss, optimizer)

            if it % args.log_interval == 0:

                if args.wandb:
                    loss.LogToWandB('loss')
                
                print("\033[4;36mEp %3d\033[0m | \033[4;32mIt %4d / %4d\033[0m | %s" %
                (epoch, bit, len(train_loader), loss.GetLogString()))
                sys.stdout.flush()

            bit += 1
            it += 1
        
        net.eval()

        '''TEST / VALIDATION'''
        correct = 0
        # for data, target in ec.test_loader:

            # data = data.to(device)
            # target = target.to(device)

            # pred = net(data)
            # correct += NumEq(pred, target)

            # pass

        acc = correct / len(test_loader.dataset) * 100.0

        if args.wandb:
            wandb.log({"accuracy": acc, "epoch": epoch})
        print("Epoch %3d - Acc %.2f" % (epoch, acc))
        sys.stdout.flush()

        # TODO: Visualize

    ##########################################################################
    ''' Report Result '''
    ##########################################################################

    with open("out.temp", 'w') as fp:
        fp.write("%f" % acc)
    print("hpt-result=%f" % acc)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
    sys.stdout.flush()
