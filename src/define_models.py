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


def dummy_autoencoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'fc_insize':32000,
    'hidden_size': 200
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    class model1(nn.Module):
        def __init__(self):
            super(model1, self).__init__()
            self.hidden = nn.Linear(p['fc_insize'], 20)
            self.out = nn.Linear(20, p['fc_insize'])
        def forward(self, X):

            X = self.hidden(X)
            X = self.out(X)
            return X


    out = model1()

    return out, p


def simple_encoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'variational':True,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)


    class simple_encoder_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim'], variational=p['variational']):
            super(simple_encoder_class, self).__init__()
            self.variational = variational
            self.fc1 = nn.Linear(16384, 10000)
            self.fc2 = nn.Linear(10000, 8000)
            self.fc3 = nn.Linear(8000, 5000)
            self.fc4 = nn.Linear(5000, 2000)
            self.fc5 = nn.Linear(2000, 1000)

            self.bn1 = nn.BatchNorm1d(1)
            self.bn2 = nn.BatchNorm1d(1)
            self.bn3 = nn.BatchNorm1d(1)
            self.bn4 = nn.BatchNorm1d(1)
            self.bn5 = nn.BatchNorm1d(1)

            self.fc6_1 = nn.Linear(1000, latent_dim)
            self.fc6_2 = nn.Linear(1000, latent_dim)

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)

        def forward(self, x):

            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.relu(self.bn4(self.fc4(x)))
            x = F.relu(self.bn5(self.fc5(x)))

            x1 = F.sigmoid(self.fc6_1(x))
            if self.variational:
                x2 = F.sigmoid(self.fc6_2(x))
                return x1, x2
            else:
                return x1, x1

    out = simple_encoder_class()

    return out, p



def simple_decoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)


    class simple_decoder_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim']):
            super(simple_decoder_class, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 1000)
            self.fc2 = nn.Linear(1000, 2000)
            self.fc3 = nn.Linear(2000, 5000)
            self.fc4 = nn.Linear(5000, 8000)
            self.fc5 = nn.Linear(8000, 10000)
            self.fc6 = nn.Linear(10000, 16384)

            self.bn5 = nn.BatchNorm1d(1)
            self.bn4 = nn.BatchNorm1d(1)
            self.bn3 = nn.BatchNorm1d(1)
            self.bn2 = nn.BatchNorm1d(1)
            self.bn1 = nn.BatchNorm1d(1)

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)


        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.relu(self.bn4(self.fc4(x)))
            x = F.relu(self.bn5(self.fc5(x)))

            x = F.tanh(self.fc6(x))

            return x

    out = simple_decoder_class()

    return out, p

def WAVE_CNN_complete_net(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'model_size': 64,
    'latent_dim': 100,
    'verbose': False,
    'num_channels': 1,
    'alpha': 0.2,
    'variational': True,
    'upsample': True,
    'post_proc_filt_len': 512,
    'drop_prob': 0.5
    }
    p = parse_parameters(p, user_parameters)
    flattened_dim = time_dim * features_dim

    class UpsampleConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
            super(UpsampleConvLayer, self).__init__()
            self.upsample = upsample
            if upsample:
                self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ConstantPad1d(reflection_padding, value = 0)
    #             self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
                self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            x_in = x
            if self.upsample:
                x_in = self.upsample_layer(x_in)
            out = self.reflection_pad(x_in)
            out = self.conv1d(out)
            return out

    class Net(nn.Module):

        def __init__(self, model_size=p['model_size'], latent_dim=p['latent_dim'],
                        verbose=p['verbose'], num_channels=p['num_channels'], alpha=p['alpha'],
                        variational=p['variational'],upsample=p['upsample'],
                        post_proc_filt_len=p['post_proc_filt_len'], drop_prob=p['drop_prob']):
            super().__init__()

            #ENCODER
            self.model_size = model_size # d
            self.num_channels = num_channels # c
            self.variational = variational
            self.alpha = alpha
            self.verbose = verbose
            self.latent_dim = latent_dim
            self.drop_prob = drop_prob
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
            self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
            self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
            self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
            self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
            self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)
            self.fc1_e = nn.Linear(256 * model_size, 1000)
            self.mu = nn.Linear(1000, latent_dim)
            self.logvar = nn.Linear(1000, latent_dim)


            self.bn1_e = nn.BatchNorm1d(model_size)
            self.bn2_e = nn.BatchNorm1d(2 * model_size)
            self.bn3_e = nn.BatchNorm1d(4 * model_size)
            self.bn4_e = nn.BatchNorm1d(8 * model_size)
            self.bn5_e = nn.BatchNorm1d(16 * model_size)

            #DECODER
            self.post_proc_filt_len = post_proc_filt_len
            self.verbose = verbose
            self.fc1 = nn.Linear(latent_dim, 256 * model_size)
            self.fc2 = nn.Linear(16384,16384)
            self.tconv1 = None
            self.tconv2 = None
            self.tconv3 = None
            self.tconv4 = None
            self.tconv5 = None
            self.upSampConv1 = None
            self.upSampConv2 = None
            self.upSampConv3 = None
            self.upSampConv4 = None
            self.upSampConv5 = None
            self.upsample = upsample
            self.fc2 = nn.Linear(16384, 16384)
            self.bn1_d = nn.BatchNorm1d(8 * model_size)
            self.bn2_d = nn.BatchNorm1d(4 * model_size)
            self.bn3_d = nn.BatchNorm1d(2 * model_size)
            self.bn4_d = nn.BatchNorm1d(model_size)
            self.bn5_d = nn.BatchNorm1d(1)



            if self.upsample:
                self.upSampConv1 = UpsampleConvLayer(16 * model_size, 8 * model_size, 25, stride=1, upsample=4)
                self.upSampConv2 = UpsampleConvLayer(8 * model_size, 4 * model_size, 25, stride=1, upsample=4)
                self.upSampConv3 = UpsampleConvLayer(4 * model_size, 2 * model_size, 25, stride=1, upsample=4)
                self.upSampConv4 = UpsampleConvLayer(2 * model_size, model_size, 25, stride=1, upsample=4)
                self.upSampConv5 = UpsampleConvLayer(model_size, num_channels, 25, stride=1, upsample=4)

            else:
                self.tconv1 = nn.ConvTranspose1d(16 * model_size, 8 * model_size, 25, stride=4, padding=11, output_padding=1)
                self.tconv2 = nn.ConvTranspose1d(8 * model_size, 4 * model_size, 25, stride=4, padding=11, output_padding=1)
                self.tconv3 = nn.ConvTranspose1d(4 * model_size, 2 * model_size, 25, stride=4, padding=11, output_padding=1)
                self.tconv4 = nn.ConvTranspose1d(2 * model_size, model_size, 25, stride=4, padding=11, output_padding=1)
                self.tconv5 = nn.ConvTranspose1d(model_size, num_channels, 25, stride=4, padding=11, output_padding=1)

            if post_proc_filt_len:
                self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)


            #KAIMING INITIALIZATION
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
                    nn.init.kaiming_normal_(m.weight.data)

        def encode(self, x):
            x = F.leaky_relu(self.bn1_e(self.conv1(x)), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.bn2_e(self.conv2(x)), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.bn3_e(self.conv3(x)), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.bn4_e(self.conv4(x)), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.bn5_e(self.conv5(x)), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            output = x.view(-1, 256 * self.model_size)
            if self.verbose:
                print(output.shape)

            return output

        def decode(self, x):

                x = self.fc1(x).view(-1, 16 * self.model_size, 16)
                x = F.relu(x)
                output = None
                if self.verbose:
                    print(x.shape)

                if self.upsample:
                    x = F.relu(self.bn1_d(self.upSampConv1(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn2_d(self.upSampConv2(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn3_d(self.upSampConv3(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn4_d(self.upSampConv4(x)))
                    if self.verbose:
                        print(x.shape)

                    output = torch.tanh(self.bn5_d(self.upSampConv5(x)))
                else:
                    x = F.relu(self.bn1_d(self.tconv1(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn2_d(self.tconv2(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn3_d(self.tconv3(x)))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.bn4_d(self.tconv4(x)))
                    if self.verbose:
                        print(x.shape)

                    output = torch.tanh(self.bn5_d(self.tconv5(x)))

                if self.verbose:
                    print(output.shape)

                if self.post_proc_filt_len:
                    # Pad for "same" filtering
                    if (self.post_proc_filt_len % 2) == 0:
                        pad_left = self.post_proc_filt_len // 2
                        pad_right = pad_left - 1
                    else:
                        pad_left = (self.post_proc_filt_len - 1) // 2
                        pad_right = pad_left
                    output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))
                    if self.verbose:
                        print(output.shape)

                return output


        def enc_func(self, x):
            #here we will be returning the logvar(log variance) and mean of our network
            #x = x.view([-1, 16384])

            x = self.encode(x)
            x = F.relu(self.fc1_e(x))
            #x = F.dropout2d(x, self.drop_prob)

            mu = F.sigmoid(self.mu(x))
            logvar = F.sigmoid(self.logvar(x))

            #mu = self.mu(x)
            #logvar = self.logvar(x)

            return mu, logvar

        def dec_func(self, z):
            #here z is the latent variable state
            z = self.decode(z)

            return z

        def reparametrize(self, mu, logvar, warm_value_reparametrize):
            #state comes from training
            #after a certain period, ad variational inference
            if self.variational:
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

def WAVE_complete_net(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)
    flattened_dim = time_dim * features_dim

    class Net(nn.Module):
        def __init__(self, num_latent=p['latent_dim']):
            super().__init__()

            #So here we will first define layers for encoder network
            self.encoder = nn.Sequential(nn.Linear(16384,2000),
                                        F.relu(nn.BatchNorm1d(2000)),

                                         nn.Linear(2000,2000),
                                         nn.BatchNorm1d(2000),
                                         nn.ReLU(),
                                         nn.Linear(2000,2000),
                                         nn.BatchNorm1d(2000),
                                         nn.ReLU(),
                                         )

            #These two layers are for getting logvar and mean
            self.fc1 = F.relu(nn.Linear(2000, 256))
            self.fc2 = F.relu(nn.Linear(256, 128))
            self.mean = nn.Linear(128, num_latent)
            self.var = nn.Linear(128, num_latent)

            #######The decoder part
            #This is the first layer for the decoder part
            self.expand = F.relu(nn.Linear(num_latent, 128))
            self.fc3 = F.relu(nn.Linear(128, 256))
            self.fc4 = F.relu(nn.Linear(256, 2000))
            self.decoder = nn.Sequential(nn.Linear(2000,2000),
                                         nn.BatchNorm1d(2000),
                                         nn.ReLU(),
                                         nn.Linear(2000,2000),
                                         nn.BatchNorm1d(2000),
                                         nn.ReLU(),
                                         nn.Linear(2000,16384)
                                         )

        def enc_func(self, x):
            #here we will be returning the logvar(log variance) and mean of our network
            x = x.view([-1, 16384])
            x = self.encoder(x)
            #x = F.dropout2d(self.fc1(x), 0.5)
            x = self.fc1(x)
            x = self.fc2(x)

            mean = self.mean(x)
            logvar = self.var(x)
            return mean, logvar

        def dec_func(self, z):
            #here z is the latent variable state
            z = self.expand(z)
            #z = F.dropout2d(self.fc3(z), 0.5)
            z = self.fc4(z)

            out = self.decoder(z)
            #out = out.view([-1, time_dim, features_dim])
            out = out.view([-1, 1, 16384])
            out = F.tanh(out)
            return out

        def reparametrize(self, mu, logvar, warm_value_reparametrize):
            #state comes from training
            #after a certain period, ad variational inference
            if self.variational:
                if dyn_variational:
                    #activated from training
                    if self.training:
                        std = torch.exp(0.5*logvar)   #So as to get std
                        noise = torch.randn_like(mu)   #So as to get the noise of standard distribution
                        #noise *= warm_value_reparametrize
                        return noise.mul(std).add_(mu)
                    else:
                        return mu
                else:
                    return mu
            else:
                return mu

        def forward(self, x):
            mu, logvar = self.enc_func(x)
            z = self.reparametrize(mean, logvar)
            out = self.dec_func(z)
            return out, mu, logvar

    out = Net()

    return out, p

def complete_net(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)
    flattened_dim = time_dim * features_dim

    class Net(nn.Module):
        def __init__(self, num_latent=p['latent_dim']):
            super().__init__()

            #So here we will first define layers for encoder network
            self.encoder = nn.Sequential(nn.Linear(flattened_dim,2000),
                                         nn.Linear(2000,2000),
                                         nn.Linear(2000,2000))

            #These two layers are for getting logvar and mean
            self.fc1 = nn.Linear(2000, 256)
            self.fc2 = nn.Linear(256, 128)
            self.mean = nn.Linear(128, num_latent)
            self.var = nn.Linear(128, num_latent)

            #######The decoder part
            #This is the first layer for the decoder part
            self.expand = nn.Linear(num_latent, 128)
            self.fc3 = nn.Linear(128, 256)
            self.fc4 = nn.Linear(256, 2000)
            self.decoder = nn.Sequential(nn.Linear(2000,2000),
                                         nn.Linear(2000,2000),
                                         nn.Linear(2000,flattened_dim))

        def enc_func(self, x):
            #here we will be returning the logvar(log variance) and mean of our network
            x = x.view([-1, flattened_dim])
            x = self.encoder(x)
            x = F.dropout2d(self.fc1(x), 0.5)
            x = self.fc2(x)

            mean = self.mean(x)
            logvar = self.var(x)
            return mean, logvar

        def dec_func(self, z):
            #here z is the latent variable state
            z = self.expand(z)
            z = F.dropout2d(self.fc3(z), 0.5)
            z = self.fc4(z)

            out = self.decoder(z)
            out = out.view([-1, time_dim, features_dim])
            out = F.sigmoid(out)
            return out

        def get_hidden(self, mean, logvar):
            if self.training:
                std = torch.exp(0.5*logvar)   #So as to get std
                noise = torch.randn_like(mean)   #So as to get the noise of standard distribution
                return noise.mul(std).add_(mean)
            else:
                return mean

        def forward(self, x):
            mean, logvar = self.enc_func(x)
            z = self.get_hidden(mean, logvar)
            out = self.dec_func(z)
            return out, mean, logvar

    out = Net()

    return out, p

def simple_encoder_spectrum(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'variational':True,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)
    flattened_dim = time_dim * features_dim

    class simple_encoder_spectrum_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim'], variational=p['variational']):
            super(simple_encoder_spectrum_class, self).__init__()
            self.variational = variational
            self.fc1 = nn.Linear(flattened_dim, 2000)
            self.fc2 = nn.Linear(2000, 2000)
            self.fc3 = nn.Linear(2000, 2000)

            self.bn1 = nn.BatchNorm1d(2000)
            self.bn2 = nn.BatchNorm1d(2000)
            self.bn3 = nn.BatchNorm1d(2000)

            self.mu = nn.Linear(2000, latent_dim)
            self.logvar = nn.Linear(2000, latent_dim)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)

        def forward(self, x):

            x = x.view(x.shape[0], time_dim * features_dim)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))


            x1 = F.sigmoid(self.mu(x))
            if self.variational:
                x2 = F.sigmoid(self.logvar(x))
                return x1, x2
            else:
                return x1, x1

    out = simple_encoder_spectrum_class()

    return out, p

def simple_decoder_spectrum(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)
    flattened_dim = time_dim * features_dim

    class simple_decoder_spectrum_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim']):
            super(simple_decoder_spectrum_class, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 2000)
            self.fc2 = nn.Linear(2000, 2000)
            self.fc3 = nn.Linear(2000, 2000)

            self.out =  nn.Linear(2000, flattened_dim)



            self.bn1 = nn.BatchNorm1d(2000)
            self.bn2 = nn.BatchNorm1d(2000)
            self.bn3 = nn.BatchNorm1d(2000)


            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)


        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))

            x = F.sigmoid(self.out(x))
            x = x.view(x.shape[0], time_dim, features_dim)

            return x

    out = simple_decoder_spectrum_class()

    return out, p

def spectrum_encoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'variational':True,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)


    class spectrum_encoder_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim'], variational=p['variational']):
            super(spectrum_encoder_class, self).__init__()
            self.variational = variational
            self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            self.fc1 = nn.Linear(256 * 6 * 6, 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, 2000)
            self.fc4 = nn.Linear(2000, 500)

            self.bn1 = nn.BatchNorm1d(4096)
            self.bn2 = nn.BatchNorm1d(4096)
            self.bn3 = nn.BatchNorm1d(2000)
            self.bn4 = nn.BatchNorm1d(500)


            self.mu = nn.Linear(500, latent_dim)
            self.logvar = nn.Linear(500, latent_dim)

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)



        def forward(self, x):

            #features extraction

            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = self.pool3(x)

            x = self.avgpool(x)

            #classification
            x = torch.flatten(x, 1)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.relu(self.bn4(self.fc4(x)))

            x1 = F.sigmoid(self.mu(x))
            if self.variational:
                x2 = F.sigmoid(self.logvar(x))
                return x1, x2
            else:
                return x1, x1

    out = spectrum_encoder_class()

    return out, p

def MNIST_encoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'variational':True,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)


    class MNIST_encoder_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim'], variational=p['variational']):
            super(MNIST_encoder_class, self).__init__()
            self.variational = variational
            self.fc1 = nn.Linear(784, 500)
            self.fc2 = nn.Linear(500, 200)

            self.bn1 = nn.BatchNorm1d(1)
            self.bn2 = nn.BatchNorm1d(1)


            self.fc3_1 = nn.Linear(200, latent_dim)
            self.fc3_2 = nn.Linear(200, latent_dim)

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)

        def forward(self, x):

            x = x.view(x.shape[0], 1, 784)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))

            x1 = F.sigmoid(self.fc3_1(x))
            if self.variational:
                x2 = F.sigmoid(self.fc3_2(x))
                return x1, x2
            else:
                return x1, x1

    out = MNIST_encoder_class()

    return out, p

def MNIST_decoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False,
    'latent_dim':100
    }
    p = parse_parameters(p, user_parameters)


    class MNIST_decoder_class(nn.Module):
        def __init__(self, latent_dim=p['latent_dim']):
            super(MNIST_decoder_class, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 200)
            self.fc2 = nn.Linear(200, 500)
            self.fc3 = nn.Linear(500, 784)

            self.bn2 = nn.BatchNorm1d(1)
            self.bn1 = nn.BatchNorm1d(1)

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)


        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.sigmoid(self.fc3(x))


            return x

    out = MNIST_decoder_class()

    return out, p

def WAVE_VAE(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':False
    }
    p = parse_parameters(p, user_parameters)

    encoder_object, dummy = WAVE_encoder(time_dim, features_dim, user_parameters=user_parameters)
    decoder_object, dummy = WAVE_decoder(time_dim, features_dim, user_parameters=user_parameters)
    reparametrize_object = reparametrize(time_dim, features_dim, user_parameters=user_parameters)

    class WAVE_VAE(nn.Module):
        def __init__(self, encoder=encoder_object, decoder=decoder_object, reparametrize=reparametrize_object):
            super(WAVE_VAE, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.reparametrize = reparametrize

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z = self.reparametrize(mu, logvar)
            reconstructed = self.decoder(z)

            return reconstructed, mu, logvar

    out = WAVE_VAE()

    return out, p
