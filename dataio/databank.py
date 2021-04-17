import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,TensorDataset
import os
import numpy as np
import random
from datetime import datetime

from .mnistm import MNISTM

from tqdm import tqdm

class CIFAR9(datasets.CIFAR10):

    def __init__(self, train, transform = None):
        super().__init__("data/cifar10-raw", download = True, train = train, transform = transform)
        self.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']
        targets = np.array(self.targets)
        idx_car = targets == 1
        idx_bird = targets == 2
        targets[idx_car] = 2
        targets[idx_bird] = 1
        idx_frog = targets == 6
        targets = targets[~idx_frog]
        idx_higher = targets > 6
        targets[idx_higher] = targets[idx_higher] - 1
        self.targets = targets.tolist()
        self.data = self.data[~idx_frog]

class STL9(datasets.STL10):

    def __init__(self, train, transform = None):
        super().__init__("data/stl10-raw", download = True, split = 'train' if train else 'test', transform = transform)
        self.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']
        labels = np.array(self.labels)
        idx_monkey = labels == 7
        labels = labels[~idx_monkey]
        idx_higher = labels > 7
        labels[idx_higher] = labels[idx_higher] - 1
        self.labels = labels.tolist()
        self.data = self.data[~idx_monkey]

# Train = synthetic; Validation = real-life
def get_VisDA18(split, size = (224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return datasets.ImageFolder(
        'data/VisDA18/visda-2018-public/openset/%s/' % split,
        transform = transform
    )

def get_custom_feat(data_dir, batch_size = 128):

    data = torch.load(data_dir)
    
    X = data['X']
    Y = data['Y']

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0],)

    meta = {'dim': X.shape[1], 'nClasses': int(max(Y).item())+1}

    return TensorDataset(X,Y), meta['nClasses']

def mnist_repeat(x):
    return x.repeat(3, 1, 1)

def dataset_bank_train (name):
    
    if name == 'mnist':
        return datasets.MNIST('data/mnist', train = True, transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(mnist_repeat)
            ]), download=True), 10

    if name == 'mnistm':
        return MNISTM('data/mnistm', train = True, transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]), download=True), 10

    if name == 'svhn':
        return datasets.SVHN('data/svhn', split = 'train', transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]), download=True), 10

    if name == 'cifar-raw':
        return CIFAR9(True, transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                # )
                transforms.Normalize((0.5,),(0.5,)),
            ])), 9

    if name == 'stl-raw':
        return STL9(True, transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=(0.43, 0.42, 0.39),
                #     std=(0.27, 0.26, 0.27)
                # )
                transforms.Normalize((0.5,),(0.5,)),
            ])), 9

    if name == 'visda-18-synthesis':
        return get_VisDA18('train', (32, 32)), 13

    if name == 'visda-18-real':
        return list(torch.utils.data.random_split(get_VisDA18('validation', (32, 32)), [50000, 5281], generator = torch.Generator().manual_seed(42)))[0], 13

    if name == 'visda-18-resnet-synthesis':
        return get_custom_feat('data/VisDA18/visda-2018-public/features/train.pkl')
    
    if name == 'visda-18-resnet-real':
        dset, nC = get_custom_feat('data/VisDA18/visda-2018-public/features/validation.pkl')
        return list(torch.utils.data.random_split(dset, [50000, 5281], generator = torch.Generator().manual_seed(42)))[0], nC

    return None

def dataset_bank_test (name):
    
    if name == 'mnist':
        return datasets.MNIST('data/mnist', train = False, transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(mnist_repeat)
            ]), download=True), 10

    if name == 'mnistm':
        return MNISTM('data/mnistm', train = False, transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]), download=True), 10

    if name == 'svhn':
        return datasets.SVHN('data/svhn', split = 'test', transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]), download=True), 10

    if name == 'cifar-raw':
        return CIFAR9(False, transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])), 9

    # TODO: Check French et al. - they resized STL to 32x32 ??
    if name == 'stl-raw':
        return STL9(False, transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])), 9

    if name == 'visda-18-synthesis':
        return get_VisDA18('train', (32, 32)), 13

    if name == 'visda-18-real':
        return list(torch.utils.data.random_split(get_VisDA18('validation', (32, 32)), [50000, 5281], generator=torch.Generator().manual_seed(42)))[1], 13
        
    if name == 'visda-18-resnet-synthesis':
        return get_custom_feat('data/VisDA18/visda-2018-public/features/train.pkl')
    
    if name == 'visda-18-resnet-real':
        dset, nC = get_custom_feat('data/VisDA18/visda-2018-public/features/validation.pkl')
        return list(torch.utils.data.random_split(dset, [50000, 5281], generator = torch.Generator().manual_seed(42)))[1], nC

    return None

def getFromDSet(dset):
    
    dl = torch.utils.data.DataLoader(dset, batch_size=1024, shuffle=False, num_workers=0)
    xs = []
    ys = []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    
    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)

    return [xs.numpy(), ys.numpy()]
