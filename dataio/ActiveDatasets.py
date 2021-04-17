import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import torch.utils.data as data
from PIL import Image

import os
import numpy as np
import errno

class ActiveSideDataset(Dataset):

    '''
    Represents a single dataset (domain) that can be queried to be semi-supervised.
    '''

    def __init__(self, data, n_classes, init_oracles):
        
        self.data = list(data)
        self.label = data[1]
        # self.data[1] = onehot(self.data[1], n_classes)

        self.n_classes = n_classes

        assert self.data[0].shape[0] == self.data[1].shape[0]

        self.size = self.data[0].shape[0]
        self.sample_size = self.size

        self.resetOracle(init_oracles)
        
    def __getitem__(self, idx):

        idx = idx if idx < self.size else random.randint(0, self.size - 1)

        desc =\
        {
            'X': torch.FloatTensor(self.data[0][idx]),
            'Y': self.data[1][idx] if self.oracle[idx] > 0 else -100, # -100 is by default ignored by torch.nn.CrossEntropyLoss
            'Y_gt': self.data[1][idx],
            'supervised': self.oracle[idx] # mask
        }

        return desc
    
    def __len__(self):
        return self.sample_size

    def setSize(self, size):
        self.sample_size = size

    def query(self, idxs):
        
        self.oracle[idxs] = 1

        # update indices
        self.supervised.update(idxs)
        for i in idxs:
            self.unsupervised.discard(i)
    
    def resetOracle(self, init_oracles):

        if init_oracles is True: # if all of us are supervised at the very beginning

            self.oracle = np.ones((self.data[1].shape[0],))
            self.supervised = set([i for i in range(self.data[1].shape[0])])
            self.unsupervised = set()

        else:
            
            self.oracle = np.zeros((self.data[1].shape[0],))
            self.supervised = set()
            self.unsupervised = set([i for i in range(self.data[1].shape[0])])
            
            # supervise some data at initialization
            if init_oracles:

                self.oracle[init_oracles] = 1

                # update indices
                self.supervised.update(init_oracles)
                for i in init_oracles:
                    self.unsupervised.discard(i)

# Ensure sampling from supervised subset at every batch for a given ratio
class SemisupervisedSampler(Sampler):

    def __init__(self, dataset, batchsize, size = None, ratio = 0.2, shuffle = True):
        self.dataset = dataset
        self.size = size or len(self.dataset)
        self.ratio = ratio
        self.bs = batchsize
        self.shuffle = shuffle
    
    def setRatio(self, ratio):
        self.ratio = ratio
        self.generateSamples()
        
    def generateSamples(self):
        count = int(self.bs * self.ratio)
        self.samples = [True] * count + [False] * (self.bs - count)
        random.shuffle(self.samples)
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        total_cnt = 0

        while total_cnt < self.size:

            self.generateSamples()

            for i in range(self.bs):

                if self.samples[i] == True and len(self.dataset.supervised) > 0:
                    yield random.sample(self.dataset.supervised, 1)[0]
                elif len(self.dataset.unsupervised) > 0:
                    yield random.sample(self.dataset.unsupervised, 1)[0]
                else:
                    yield random.randint(0, len(self.dataset) - 1)
            
                total_cnt += 1
                
                if total_cnt >= self.size:
                    break

class GeneralSampler(Sampler):

    def __init__(self, getter, batchsize, length):

        self.getter = getter
        self.length = length
        self.batchsize = batchsize

    def __len__(self):

        return self.length

    def __iter__(self):

        total_cnt = 0
        idx_range = self.getter()
        #print(idx_range)

        if len(idx_range) == 0:
            idx_range = [0]

        while total_cnt < self.length:

            for i in range(self.batchsize):

                # TODO: use repeated till last dataset loop
                ix = random.choice(idx_range)
                #print(ix)
                yield ix

                total_cnt += 1
                if total_cnt >= self.length:
                    break

def ATDataset_collateFn(batch):

    batch_desc = {}
    
    for item in batch:
        for key in item:
            if key not in batch_desc:
                batch_desc[key] = []
            batch_desc[key].append(item[key])

    # combine GLOFEA tensors (ResNet, raw word embeddings, etc.)
    for key in batch_desc:
        if type(batch_desc[key][0]) is torch.Tensor and (batch_desc[key][0].dtype == torch.float32 or batch_desc[key][0].dtype == torch.int64):
            batch_desc[key] = torch.stack(batch_desc[key], dim = 0)
        else:
            batch_desc[key] = torch.LongTensor([k for k in batch_desc[key]])

    return batch_desc

# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703
def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses
    for item in dataset:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(dataset)
    for idx in range(len(dataset)):
        weight[idx] = weight_per_class[dataset[idx]]
    return weight

def getSampler(dataset, nclasses, datalen):
    weights = make_weights_for_balanced_classes(dataset, nclasses)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datalen or len(weights))
    return sampler

# TODO: fix this to have the expected behaviour
class DatasetWrapper():

    '''
    Fixed streagy - 
        Sampler needs fixed sample sequence decided at the beginning of each epoch.
        Rotate through the entire dataset till no full-set can fit, then pick randomly without replacement to fill the entire sequence.

        Sample regular minibatch from source
        Sample regular minibatch from labelled target (t_supRatio fixed at some constant, say 0.5)
        Sample regular minibatch from unlabelled target
    '''

    def __init__(self, s_dataset, t_dataset, batch_size = 64, num_workers = 0):

        assert s_dataset.n_classes == t_dataset.n_classes

        self.s_dataset = s_dataset
        self.t_dataset = t_dataset
        self.slabel = s_dataset.label
        self.tlabel = t_dataset.label
        self.source = s_dataset.data
        self.target = t_dataset.data
        self.n_classes = s_dataset.n_classes

        self.oracle = t_dataset.oracle
        self.supervised = t_dataset.supervised
        self.unsupervised = t_dataset.unsupervised

        self.size = max(len(self.s_dataset), len(self.t_dataset))

        self.s_sampler = GeneralSampler(lambda : list(range(len(self.s_dataset))), batch_size, self.size)
        self.t_sampler = GeneralSampler(lambda : list(self.t_dataset.unsupervised), batch_size, self.size)
        self.q_sampler = GeneralSampler(lambda : list(self.t_dataset.supervised), batch_size, self.size)

        self.s_loader = DataLoader(s_dataset, batch_size, num_workers = num_workers, collate_fn = ATDataset_collateFn, sampler = self.s_sampler, drop_last = True)
        self.t_loader = DataLoader(t_dataset, batch_size, num_workers = num_workers, collate_fn = ATDataset_collateFn, sampler = self.t_sampler, drop_last = True)
        self.q_loader = DataLoader(t_dataset, batch_size, num_workers = num_workers, collate_fn = ATDataset_collateFn, sampler = self.q_sampler, drop_last = True)

        self.s_iter = iter(self.s_loader)
        self.t_iter = iter(self.t_loader)
        self.q_iter = iter(self.q_loader)

    def __iter__(self):
        return self
    
    def RefreshSamplers(self):
        self.s_iter = iter(self.s_loader)
        self.t_iter = iter(self.t_loader)
        self.q_iter = iter(self.q_loader)

    def __next__(self):

        try:
            s = next(self.s_iter)
            t = next(self.t_iter)
            q = next(self.q_iter)

            return {
                'srcX': s['X'],
                'srcY': s['Y'],
                'dstX': t['X'],
                'dstY': t['Y'],
                'dstY_gt': t['Y_gt'],
                'qX': q['X'],
                'qY': q['Y']
            }
        
        except StopIteration as e:
            self.s_iter = iter(self.s_loader)
            self.t_iter = iter(self.t_loader)
            self.q_iter = iter(self.q_loader)
            raise StopIteration

    def __len__(self):
        return self.size

    def resetOracle(self, inito):
        self.t_dataset.resetOracle(inito)
        self.oracle = self.t_dataset.oracle
        self.supervised = self.t_dataset.supervised
        self.unsupervised = self.t_dataset.unsupervised
