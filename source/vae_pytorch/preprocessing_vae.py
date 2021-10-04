from __future__ import print_function
import loadconfig
import configparser
import random
import numpy as np
import librosa
import os, sys
from modules import *
import utility_functions as uf

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SRNN_DATA_PATH = os.path.abspath(cfg.get('samplernn', 'samplernn_data_path'))
SR = cfg.getint('main', 'main_sr')
ENV_LENGTH = cfg.getint('vae', 'env_length_preprocessing')
OUTPUT_FOLDER = cfg.get('vae', 'output_folder_preprocessing')
SEQUENCE_LENGTH = cfg.getint('vae', 'sequence_length_preprocessing')

features_extractor = Preprocessing(sr=SR, env_length=ENV_LENGTH)

dataset = []

#build matrix with all preprocessed data
print ('preprocessing')
categories = list(filter(lambda x: '.DS_Store' not in x,os.listdir(SRNN_DATA_PATH)))
for cat in categories:  #iterate categories
    print ('\ncategory: ' + str(cat))
    cat_path = os.path.abspath(os.path.join(SRNN_DATA_PATH, cat))
    curr_models = list(filter(lambda x: '.DS_Store' not in x, os.listdir(cat_path)))  #iterate models
    for mod in curr_models: #iterate models
        print ('\nmodel: ' + str(mod))
        mod_path = os.path.abspath(os.path.join(cat_path, mod))
        sounds_path = os.path.abspath(os.path.join(mod_path, 'sounds', 'dur_' + str(SEQUENCE_LENGTH)))
        if os.path.exists(sounds_path): #id sounds are there
            target_paths = list(filter(lambda x: '.DS_Store' not in x, os.listdir(sounds_path)))
            for var in target_paths:  #iterate variations
                if var == 'model_0':  #USE ONLY BEST MODELS' SOUNDS
                    print ('\nvariation: ' + str(var))
                    var_path = os.path.abspath(os.path.join(sounds_path, var))
                    final_sounds = list(filter(lambda x: '.DS_Store' not in x, os.listdir(var_path)))
                    num_sounds = len(final_sounds)
                    index = 0
                    for s in final_sounds:
                        sound = os.path.abspath(os.path.join(var_path, s))
                        features = features_extractor.extract_envs(sound)
                        dataset.append(features)
                        uf.print_bar(index, num_sounds)
                        index += 1


#save dataset
dataset = np.array(dataset)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
output_name = os.path.abspath(os.path.join(OUTPUT_FOLDER, 'dataset.npy'))
np.save(output_name, dataset)
