import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms
from audtorch import metrics
import numpy as np
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

batch_size = cfg.getint('training_defaults', 'batch_size')


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

def warm_up(epochs, init_silence=1500, perc=0.0001, tot_epochs=500):
    pad = np.zeros(epochs)
    #ramp_time = int(epochs*perc) - init_silence
    ramp_time = tot_epochs
    start = init_silence
    end = init_silence + ramp_time
    ramp = np.arange(ramp_time) / ramp_time
    pad[start:end] = ramp
    pad [end:] = 1.

    return pad

def warm_up_reparametrize(epochs, init_silence=1000, perc=0.0001, tot_epochs=500):
    pad = np.zeros(epochs)
    #ramp_time = int(epochs*perc) - init_silence
    ramp_time = tot_epochs
    start = init_silence
    end = init_silence + ramp_time
    ramp = np.arange(ramp_time) / ramp_time
    pad[start:end] = ramp
    pad [end:] = 1.

    return pad

def loss_KLD(mu, logvar, epoch, warm_ramp, recon_x, kld_weight=1.):

    if warm_up:
        kld_weight_epoch = kld_weight * warm_ramp[epoch]
    else:
        kld_weight_epoch = kld_weight

    '''
    KLD = kld_weight_epoch * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    #KLD = KLD.sum(1).mean(0, True)
    '''
    #scaling_factor = recon_x.shape[0]*recon_x.shape[1]*recon_x.shape[2]
    scaling_factor = recon_x.shape[0]

    ####Now we are gonna define the KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * kld_weight_epoch * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
    kl_loss /= scaling_factor

    return kl_loss


def loss_recon(recon_x, x, features_type):

    '''
    #x = x.view(x.shape[0], 1, 784)
    if features_type == 'waveform':
        recon_loss = 1 -  torch.abs(CCC_loss(recon_x, x))
    elif features_type == 'spectrum':
        recon_loss = F.binary_cross_entropy(recon_x, x)
        recon_loss /= batch_size

    #recon_mean_distance = torch.abs(CCC_loss(recon_x, mean_target))
    '''
    if features_type == 'waveform':
        recon_loss = 1 -  torch.abs(CCC_loss(recon_x, x))
    elif features_type == 'spectrum':
        category1 = nn.BCELoss()
        recon_loss = category1(recon_x, x)

    return recon_loss

def loss_joint(recon_x, x, mu, logvar, epoch, warm_ramp, features_type, kld_weight=-0.5):

    '''
    #recon_loss = torch.sum(F.mse_loss(recon_x, x, reduction='none'))
    #recon_loss /= recon_x.shape[-1]
    #recon_x_0to1 = torch.add(torch.mul(recon_x, 0.5), 0.5)
    #x_0to1 = torch.add(torch.mul(x, 0.5), 0.5)
    #recon_loss = F.binary_cross_entropy(recon_x_0to1, x_0to1)
    #recon_loss /= batch_size
    #recon_loss = torch.log(loss_function_decoder(recon_x, x))
    recon_loss = loss_recon(recon_x, x, features_type)

    KLD = loss_KLD(mu, logvar, epoch, warm_ramp)
    #joint_loss = recon_loss

    #mean_target_distance = distance_from_mean(recon_x, mean_target)

    joint_loss = recon_loss + KLD
    #joint_loss = recon_loss + KLD - mean_target_distance
    kl_loss /= scaling_factor
    '''

    #recon_loss = loss_recon(recon_x, x, features_type)
    scaling_factor = recon_x.shape[0]
    recon_loss = torch.sum(F.mse_loss(recon_x, x, reduction='none'))
    recon_loss /= scaling_factor
    kl_loss = loss_KLD(mu, logvar, epoch, warm_ramp, recon_x, kld_weight)

    joint_loss = recon_loss + kl_loss

    return joint_loss
