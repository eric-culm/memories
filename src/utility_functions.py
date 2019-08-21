from __future__ import print_function
import numpy as np
import math, copy
import os
from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import iirfilter, butter, filtfilt, lfilter
from shutil import copyfile
import librosa
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SR = cfg.getint('sampling', 'sr_target')

tol = 1e-14    # threshold used to compute phase

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def isPower2(num):
    #taken from Xavier Serra's sms tools
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0

def wavread(file_name):
    #taken from Xavier Serra's sms tools
    '''
    read wav file and converts it from int16 to float32
    '''
    sr, samples = read(file_name)
    samples = np.float32(samples)/norm_fact[samples.dtype.name] #float conversion

    return sr, samples

def wavwrite(y, fs, filename):
    #taken from Xavier Serra's sms tools
    """
    Write a sound file from an array with the sound and the sampling rate
    y: floating point array of one dimension, fs: sampling rate
    filename: name of file to create
    """
    x = copy.deepcopy(y)                         # copy array
    x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
    x = np.int16(x)                              # converting to int16 type
    write(filename, fs, x)

def print_bar(index, total):
    perc = int(index / total * 20)
    perc_progress = int(np.round((float(index)/total) * 100))
    inv_perc = int(20 - perc - 1)
    strings = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
    print ('\r', strings, end='')

def folds_generator(num_folds, foldable_list, percs):
    '''
    create dict with a key for every actor (or foldable idem)
    in each key are contained which actors to put in train, val and test
    '''
    tr_perc = percs[0]
    val_perc = percs[1]
    test_perc = percs[2]
    num_actors = len(foldable_list)
    ac_list = foldable_list * num_folds
    n_train = int(np.round(num_actors * tr_perc))
    n_val = int(np.round(num_actors * val_perc))
    n_test = int(num_actors - (n_train + n_val))
    shift = num_actors / num_folds
    fold_actors_list = {}
    for i in range(num_folds):
        curr_shift = int(shift * i)
        tr_ac = ac_list[curr_shift:curr_shift+n_train]
        val_ac = ac_list[curr_shift+n_train:curr_shift+n_train+n_val]
        test_ac = ac_list[curr_shift+n_train+n_val:curr_shift+n_train+n_val+n_test]
        fold_actors_list[i] = {'train': tr_ac,
                          'val': val_ac,
                          'test': test_ac}

    return fold_actors_list

def build_matrix_dataset(merged_predictors, actors_list):
    '''
    load preprocessing dict and output numpy matrices of predictors and target
    containing only samples defined in actors_list
    '''
    predictors = np.array([])
    target = np.array([])
    index = 0
    total = len(actors_list)
    for i in actors_list:
        if i == actors_list[0]:  #if is first item
            predictors = np.array(merged_predictors[i])
            #target = np.array(merged_target[i],dtype='float32')
            #print (i, predictors.shape)
        else:
            if np.array(merged_predictors[i]).shape != (0,):  #if it not void due to preprocessing errors
                predictors = np.vstack((predictors, np.array(merged_predictors[i])))
                #target = np.concatenate((target, np.array(merged_target[i],dtype='float32')), axis=0)
        index += 1
        perc = int(index / total * 20)
        perc_progress = int(np.round((float(index)/total) * 100))
        inv_perc = int(20 - perc - 1)
        string = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
        print ('\r', string, end='')
    print(' | shape: ' + str(predictors.shape))
    print ('\n')

    return predictors

def find_longest_audio(input_folder):
    '''
    look for all .wav files in a folder and
    return the duration (in samples) of the longest one
    '''
    contents = os.listdir(input_folder)
    file_sizes = []
    for file in contents:
        if file[-3:] == "wav": #selects just wav files
            file_name = input_folder + '/' + file   #construct file_name string
            try:
                samples, sr = librosa.core.load(file_name, sr=None)  #read audio file
                #samples = strip_silence(samples)
                file_sizes.append(len(samples))
            except ValueError:
                pass
    max_file_length = max(file_sizes)
    max_file_length = (max_file_length + 10 )/ float(sr)

    return max_file_length, sr

def strip_silence(input_vector, threshold=35):
    split_vec = librosa.effects.split(input_vector, top_db = threshold)
    onset = split_vec[0][0]
    offset = split_vec[-1][-1]
    cut = input_vector[onset:offset]

    return cut

def preemphasis(input_vector, fs):
    '''
    2 simple high pass FIR filters in cascade to emphasize high frequencies
    and cut unwanted low-frequencies
    '''
    #first gentle high pass
    alpha=0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero,past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present,past)
    #second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.) # Normalize the frequency
    b, a = butter(8, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output

def onehot(value, range):
    '''
    int to one hot vector conversion
    '''
    one_hot = np.zeros(range)
    one_hot[value] = 1

    return one_hot
