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

def WAVE_decoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    GENERATOR: from latent dim to 1-sec sound
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':True,
    'model_size': 64,
    'upsample': True
    }
    print ('culo')
    print (p)

    p = parse_parameters(p, user_parameters)
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

            #always return model AND p!!!
    class WAVE_decoder_class(nn.Module):
        def __init__(self, model_size=p['model_size'], ngpus=1, num_channels=1, latent_dim=100,
                    post_proc_filt_len=512, verbose=p['verbose'], upsample=p['upsample']):
            super(WAVE_decoder_class, self).__init__()
            self.ngpus = ngpus
            self.model_size = model_size # d
            self.num_channels = num_channels # c
            self.latent_dim = latent_dim
            self.post_proc_filt_len = post_proc_filt_len
            self.verbose = verbose
            self.fc1 = nn.Linear(latent_dim, 256 * model_size)
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

            for m in self.modules():
                if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)

        def forward(self, x):

                x = self.fc1(x).view(-1, 16 * self.model_size, 16)
                x = F.relu(x)
                output = None
                if self.verbose:
                    print(x.shape)

                if self.upsample:
                    x = F.relu(self.upSampConv1(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.upSampConv2(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.upSampConv3(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.upSampConv4(x))
                    if self.verbose:
                        print(x.shape)

                    output = torch.tanh(self.upSampConv5(x))
                else:
                    x = F.relu(self.tconv1(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.tconv2(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.tconv3(x))
                    if self.verbose:
                        print(x.shape)

                    x = F.relu(self.tconv4(x))
                    if self.verbose:
                        print(x.shape)

                    output = torch.tanh(self.tconv5(x))

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



    out = WAVE_decoder_class()

    return out, p


def WAVE_encoder(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':True,
    'model_size':64
    }
    p = parse_parameters(p, user_parameters)

    class WAVE_encoder_class(nn.Module):
        def __init__(self, model_size=p['model_size'], num_channels=1, shift_factor=2, alpha=0.2, verbose=p['verbose'], latent_size=100):
            super(WAVE_encoder_class, self).__init__()
            self.model_size = model_size # d
            self.num_channels = num_channels # c
            self.shift_factor = shift_factor # n
            self.alpha = alpha
            self.verbose = verbose
            self.latent_size = latent_size
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
            self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
            self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
            self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
            self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
            self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)
            self.fc1_1 = nn.Linear(256 * model_size, latent_size)
            self.fc1_2 = nn.Linear(256 * model_size, latent_size)

            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)

        def forward(self, x):
            x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
            if self.verbose:
                print(x.shape)

            x = x.view(-1, 256 * self.model_size)
            if self.verbose:
                print(x.shape)

            mu = torch.sigmoid(self.fc1_1(x))
            logvar = torch.sigmoid(self.fc1_2(x))

            return mu, logvar

    out = WAVE_encoder_class()

    return out, p

def reparametrize(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    RICORDARE!!!! CHE IN INFERENCE BUTTA FUORI LA MEDIA
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'verbose':True
    }
    p = parse_parameters(p, user_parameters)

    class reparametrize(nn.Module):
        def __init__(self):
            super(reparametrize, self).__init__()
            #nothing

        def forward(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = std.data.new(std.size()).normal_()

                return eps.mul(std).add_(mu)

            else:
                # During inference, we simply spit out the mean of the
                # learned distribution for the current input.  We could
                # use a random sample from the distribution, but mu of
                # course has the highest probability.
                return mu


    out = reparametrize()

    return out

def simple_encoder(time_dim, features_dim, user_parameters=['niente = 0']):
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


    class simple_encoder_class(nn.Module):
        def __init__(self, latent_dim=100):
            super(simple_encoder_class, self).__init__()
            self.fc1 = nn.Linear(16384, 10000)
            self.fc2 = nn.Linear(10000, 8000)
            self.fc3 = nn.Linear(8000, 5000)
            self.fc4 = nn.Linear(5000, 2000)
            self.fc5 = nn.Linear(2000, 1000)
            self.fc6_1 = nn.Linear(1000, latent_dim)
            self.fc6_2 = nn.Linear(1000, latent_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x1 = F.sigmoid(self.fc6_1(x))
            x2 = F.sigmoid(self.fc6_2(x))

            return x1, x2

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
    'verbose':False
    }
    p = parse_parameters(p, user_parameters)


    class simple_decoder_class(nn.Module):
        def __init__(self, latent_dim=100):
            super(simple_decoder_class, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 1000)
            self.fc2 = nn.Linear(1000, 2000)
            self.fc3 = nn.Linear(2000, 5000)
            self.fc4 = nn.Linear(5000, 8000)
            self.fc5 = nn.Linear(8000, 10000)
            self.fc6 = nn.Linear(10000, 16384)


        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.tanh(self.fc6(x))

            return x

    out = simple_decoder_class()

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
