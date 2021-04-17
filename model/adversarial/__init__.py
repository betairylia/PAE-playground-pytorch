from .VanillaGAN import VanillaGANdis
from .WassersteinGAN_GP import WGANdis

def GetAdvModel(modelname, input_dim, use_batchnorm = False, hidden_dim = 512, GP_lambda = 10.0):

    if modelname == 'vanilla':
        return VanillaGANdis(input_dim, hidden_dim)

    elif modelname == 'wasserstein_gp':
        return WGANdis(use_batchnorm, input_dim, hidden_dim, GP_lambda, GP_samples = 1)

    else:
        raise ValueError("Adversarial Model %s unknown !" % modelname)
