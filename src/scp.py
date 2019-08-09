#CONVOLUTIONAL NEURAL NETWORK
#tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification
from __future__ import print_function
import subprocess
from shutil import copyfile
import loadconfig
import configparser

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
a = 0
copy_folder = '../sshcopy/'
#load parameters from config file
'''
TORCH_SAVE_MODEL = cfg.get('model', 'torch_save_model')
TRAINING_PREDICTORS = cfg.get('model', 'training_predictors_load')
TRAINING_TARGET = cfg.get('model', 'training_target_load')
VALIDATION_PREDICTORS = cfg.get('model', 'validation_predictors_load')
VALIDATION_TARGET = cfg.get('model', 'validation_target_load')
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')

def copy_new_model_toclipboard():
    dst = copy_folder + TORCH_SAVE_MODEL.split('/')[-1]
    copyfile(TORCH_SAVE_MODEL, dst)
'''

def s2l():
    inpath = '~/copy/copy/*.xls'
    outpath = '../temp'
    string = 'scp -r aczk407@fairlight.nsqdc.city.ac.uk:' + inpath + ' ' + outpath
    subprocess.Popen(string, shell=True)
