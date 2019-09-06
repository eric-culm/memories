import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms
from audtorch import metrics
import numpy as np

CCC_loss =  metrics.ConcordanceCC()


def loss_function_joint_old(recon_x, x, mu, logvar, epoch):
    # how well do input x and output recon_x agree?
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))  #original from paper
    recon_x_0to1 = torch.add(torch.mul(recon_x, 0.5), 0.5)
    x_0to1 = torch.add(torch.mul(x, 0.5), 0.5)

    #recon_loss = F.binary_cross_entropy(recon_x_0to1, x_0to1)
    recon_loss = torch.sum(F.mse_loss(recon_x, x, reduction='none'))
    recon_loss /= recon_x.shape[-1]

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    '''
    # Normalise by same number of elements as in reconstruction
    KLD /= p['batch_size'] * time_dim * features_dim
    '''


    recon_loss /= batch_size
    KLD /= batch_size
    #print ('')
    #print (recon_loss, KLD)

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian

    #return recon_loss + KLD
    return recon_loss

mean_target = torch.zeros(16384)

def warm_up(epochs, init_silence=0, perc=0.0001):
    pad = np.zeros(epochs)
    ramp_time = int(epochs*perc) - init_silence
    start = init_silence
    end = init_silence + ramp_time
    ramp = np.arange(ramp_time) / ramp_time
    pad[start:end] = ramp
    pad [end:] = 1.

    return pad

def loss_KLD(mu, logvar, epoch, warm_ramp, kld_weight=-0.5):

    if warm_up:
        kld_weight_epoch = kld_weight * warm_ramp[epoch]
    else:
        kld_weight_epoch = kld_weight


    KLD = kld_weight_epoch * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    #KLD = KLD.sum(1).mean(0, True)

    return KLD


def loss_recon(recon_x, x):

    x = x.view(x.shape[0], 1, 784)
    recon_loss = 1 -  torch.abs(CCC_loss(recon_x, x))
    #recon_loss = F.binary_cross_entropy(recon_x, x.view(x.shape[0],1, 784), reduction='sum')

    #recon_mean_distance = torch.abs(CCC_loss(recon_x, mean_target))

    return recon_loss

def distance_from_mean(recon_x, mean_distribution):
        recon_x = recon_x.double()
        mean_distribution = mean_distribution.double()
        mean_distance = 1 -  torch.abs(CCC_loss(recon_x, mean_distribution))
        return mean_distance.float()

def loss_joint(recon_x, x, mu, logvar, epoch, warm_ramp, mean_target, kld_weight=-0.5):


    #recon_loss = torch.sum(F.mse_loss(recon_x, x, reduction='none'))
    #recon_loss /= recon_x.shape[-1]
    #recon_x_0to1 = torch.add(torch.mul(recon_x, 0.5), 0.5)
    #x_0to1 = torch.add(torch.mul(x, 0.5), 0.5)
    #recon_loss = F.binary_cross_entropy(recon_x_0to1, x_0to1)
    #recon_loss /= batch_size
    #recon_loss = torch.log(loss_function_decoder(recon_x, x))
    #recon_loss = loss_recon(recon_x, x)
    recon_loss = F.binary_cross_entropy(recon_x, x.view(x.shape[0],1, 784), reduction='sum')

    KLD = loss_KLD(mu, logvar, epoch, warm_ramp)
    #joint_loss = recon_loss

    #mean_target_distance = distance_from_mean(recon_x, mean_target)

    joint_loss = recon_loss + KLD
    #joint_loss = recon_loss + KLD - mean_target_distance

    return joint_loss
