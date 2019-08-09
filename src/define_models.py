from __future__ import print_function
import numpy as np
import configparser
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils import np_utils
from keras.backend import int_shape
from keras import regularizers
from keras import optimizers
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

def simple_CNN(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [16, 12],
    'pool_size': [2,2],
    'conv1_depth': 10,
    'hidden_size': 100}

    reg = regularizers.l2(p['regularization_lambda'])

    #THEN CALL PARSE_PAREMETERS TO OVERWRITE DEFAULT PARAMETERS
    #WITH PARAMETERS DEFINED IN THE UI SCRIPT
    p = parse_parameters(p, user_parameters)

    #FINALLY DECLARE YOUR ARCHITECTURE AND RETURN THE MODEL
    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict
    conv_1 = Convolution2D(p['conv1_depth'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_1)
    flat = Flatten()(pool_1)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(flat)
    out = Dense(8, activation='softmax')(hidden)
    model = Model(inputs=input_data, outputs=out)

    #always return model AND p!!!
    return model, p

def EXAMPLE_model_regression(time_dim, features_dim, user_parameters=['niente = 0']):
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
    out = Dense(1, activation='linear')(hidden)
    model = Model(inputs=input_data, outputs=out)

    #always return model AND p!!!
    return model, p


if __name__ == '__main__':
    main()
