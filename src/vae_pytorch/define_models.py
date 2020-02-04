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

def parse_parameters(defaults, parameters):
    for param in parameters:
        param = param.split('=')
        item = param[0].replace(' ', '')
        value = eval(param[1].replace(' ', ''))
        defaults[item] = value
    return defaults


#DEFINE HERE YOUR MODELS!!

def EXAMPLE_model_classification(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [16, 12],
    'kernel_size_2': [8, 12],
    'kernel_size_3': [5,7],
    'pool_size': [2,2],
    'conv1_depth': 20,
    'conv2_depth': 28,
    'conv3_depth': 40,
    'drop_prob': 0.3,
    'hidden_size': 200}

    reg = regularizers.l2(p['regularization_lambda'])

    #THEN CALL PARSE_PAREMETERS TO OVERWRITE DEFAULT PARAMETERS
    #WITH PARAMETERS DEFINED IN THE UI SCRIPT
    p = parse_parameters(p, user_parameters)

    #FINALLY DECLARE YOUR ARCHITECTURE AND RETURN THE MODEL
    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict
    conv_1 = Convolution2D(p['conv1_depth'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_1)
    conv_2 = Convolution2D(p['conv2_depth'], (p['kernel_size_2'][0],p['kernel_size_2'][1]), padding='same', activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_2)
    conv_3 = Convolution2D(p['conv3_depth'], (p['kernel_size_3'][0],p['kernel_size_3'][1]), padding='same', activation='tanh')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_3)
    flat = Flatten()(pool_3)
    drop_1 = Dropout(p['drop_prob'])(flat)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(drop_1)
    out = Dense(8, activation='softmax')(hidden)
    model = Model(inputs=input_data, outputs=out)

    #always return model AND p!!!
    return model, p



def WAVE_complete_net(time_dim, features_dim, user_parameters=['niente = 0']):
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
    'input_dim' : 2000
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

            #So here we will first define layers for encoder network
            self.encoder = nn.Sequential(nn.Linear(input_dim,input_dim/2),
                                        nn.BatchNorm1d(input_dim/2)),
                                         nn.ReLU(),
                                         nn.Linear(input_dim/2),input_dim/4)),
                                         nn.BatchNorm1d(input_dim/4)),
                                         nn.ReLU(),
                                         nn.Linear(input_dim/4),input_dim/8)),
                                         nn.BatchNorm1d(input_dim/8)),
                                         nn.ReLU(),
                                         )

            #These two layers are for getting logvar and mean
            self.fc1 = nn.Linear(input_dim/8, input_dim/16)
            self.fc2 = nn.Linear(input_dim/16, input_dim/32)
            self.mean = nn.Linear(input_dim/32, latent_dim)
            self.var = nn.Linear(input_dim/32, latent_dim)

            #######The decoder part
            #This is the first layer for the decoder part
            self.expand = nn.Linear(latent_dim, input_dim/32)
            self.fc3 = nn.Linear(input_dim/32, input_dim/16)
            self.fc4 = nn.Linear(input_dim/16, input_dim/8)
            self.decoder = nn.Sequential(nn.Linear(input_dim/8,input_dim/4),
                                         nn.BatchNorm1d(input_dim/4),
                                         nn.ReLU(),
                                         nn.Linear(input_dim/4,input_dim/2),
                                         nn.BatchNorm1d(input_dim/2),
                                         nn.ReLU(),
                                         nn.Linear(input_dim/2,input_dim)
                                         )

        def enc_func(self, x):
            #here we will be returning the logvar(log variance) and mean of our network
            x = x.view([-1, input_dim])
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
            out = out.view([-1, 1, input_dim])
            out = torch.tanh(out)
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
