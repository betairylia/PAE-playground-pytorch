from .basic import BinaryLossOp, LossComposition
from .adversarial import VanillaAdversarialLoss, WassersteinAdversarialLoss, GradientPenalty
from .VAT import VATLoss, ShannonEntropy
from .VMT import VMTLoss

from . import PyTorchOT as torchOT
from .sinkhorn import SinkhornLoss
