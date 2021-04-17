from .DTN import DTN
from .MLP import PlainMLP
from .ResNet_ImageNet import ResNet_wrapper as ResNet
from .smallCNN import SmallGAPConv as SmallCNN
from .adversarial import GetAdvModel
from .VMTmodels import VMTModel

from . import particle

def GetMainModel(modelname, nOut, use_batchnorm=True, input_instanceNorm=False):
    
    if modelname == 'DTN':
        return DTN(use_batchnorm, input_instanceNorm)

    elif modelname == 'ResNet':
        return ResNet(use_batchnorm, input_instanceNorm)

    elif modelname == 'smallCNN':
        return SmallCNN(use_batchnorm, input_instanceNorm, nOut)

    elif modelname == 'smallVMTCNN':
        return VMTModel(False, use_batchnorm, input_instanceNorm, nOut)

    elif modelname == 'largeVMTCNN':
        return VMTModel(True, use_batchnorm, input_instanceNorm, nOut)

    else:
        raise ValueError("Model %s unknown !" % modelname)

def GetParticleModel(modelname):
    
    if modelname == 'test':
        return particle.TestModel()

    else:
        raise ValueError("Model %s unknown !" % modelname)
