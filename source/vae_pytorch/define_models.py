from __future__ import print_function
import numpy as np
import configparser
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import sys

#DEFINE HERE YOUR MODELS!!

def parse_parameters(defaults, parameters):
    '''
    join all requested constrains in one dict.
    constrains_list is a list of dicts
    '''
    for key in parameters.keys():
        curr_value = parameters[key]
        defaults[key] = curr_value
    return defaults

def WAVE_complete_net(user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':10,
    'dropout': False,
    'drop_prob': 0.5,
    'input_dim' : 2000,
    'variational': True
    }
    p = parse_parameters(p, user_parameters)

    class Net(nn.Module):
        def __init__(self, latent_dim=p['latent_dim'], variational=p['variational'],
                    dropout = p['dropout'], drop_prob=p['drop_prob'], input_dim=p['input_dim']):
            super().__init__()
            self.variational = variational
            self.dropout = dropout
            self.drop_prob = drop_prob
            self.latent_dim = latent_dim
            self.input_dim = input_dim

            #So here we will first define layers for encoder network
            self.encoder = nn.Sequential(nn.Linear(self.input_dim,self.input_dim//2),
                                        nn.BatchNorm1d(self.input_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(self.input_dim//2,self.input_dim//4),
                                        nn.BatchNorm1d(self.input_dim//4),
                                        nn.ReLU(),
                                        nn.Linear(self.input_dim//4,self.input_dim//8),
                                        nn.BatchNorm1d(self.input_dim//8),
                                        nn.ReLU(),
                                        )

            #These two layers are for getting logvar and mean
            self.fc1 = nn.Linear(self.input_dim//8, self.input_dim//16)
            self.fc2 = nn.Linear(self.input_dim//16, self.input_dim//32)
            self.mean = nn.Linear(self.input_dim//32, latent_dim)
            self.var = nn.Linear(self.input_dim//32, latent_dim)

            #######The decoder part
            #This is the first layer for the decoder part
            self.expand = nn.Linear(latent_dim, self.input_dim//32)
            self.fc3 = nn.Linear(self.input_dim//32, self.input_dim//16)
            self.fc4 = nn.Linear(self.input_dim//16, self.input_dim//8)
            self.decoder = nn.Sequential(nn.Linear(self.input_dim//8,self.input_dim//4),
                                         nn.BatchNorm1d(self.input_dim//4),
                                         nn.ReLU(),
                                         nn.Linear(self.input_dim//4,self.input_dim//2),
                                         nn.BatchNorm1d(self.input_dim//2),
                                         nn.ReLU(),
                                         nn.Linear(self.input_dim//2,self.input_dim)
                                         )

        def enc_func(self, x):
            #here we will be returning the logvar(log variance) and mean of our network
            x = x.view([-1, self.input_dim])
            x = self.encoder(x)
            if self.dropout:
                x = F.dropout2d(x, self.drop_prob)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            mean = torch.sigmoid(self.mean(x))
            logvar = torch.sigmoid(self.var(x))
            return mean, logvar

        def dec_func(self, z):
            #here z is the latent variable state
            z = F.relu(self.expand(z))
            z = F.relu(self.fc3(z))
            z = F.relu(self.fc4(z))
            if self.dropout:
                z = F.dropout2d(z, self.drop_prob)

            out = self.decoder(z)
            out = out.view([-1, 1, self.input_dim])
            out = torch.sigmoid(out)
            return out

        def reparametrize(self, mu, logvar, warm_value_reparametrize):
            #state comes from training
            #after a certain period, ad variational inference
            if self.variational:
                #activated from training
                if self.training:
                    std = torch.exp(0.5*logvar)   #So as to get std
                    noise = torch.randn_like(mu)   #So as to get the noise of standard distribution
                    noise *= warm_value_reparametrize
                    return noise.mul(std).add_(mu)
                else:
                    return mu
            else:
                return mu

        def forward(self, x, warm_value_reparametrize):
            mu, logvar = self.enc_func(x)
            z = self.reparametrize(mu, logvar, warm_value_reparametrize)
            out = self.dec_func(z)
            return out, mu, logvar

    out = Net()

    return out, p
