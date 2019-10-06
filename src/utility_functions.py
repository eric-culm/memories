from __future__ import print_function
import numpy as np
import math, copy
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import iirfilter, butter, filtfilt, lfilter
from shutil import copyfile
from sklearn.manifold import TSNE
import torch
import scipy
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

    #ensure that no set has 0 actors
    if n_test == 0 or n_val == 0:
        n_test = int(np.ceil(num_actors*test_perc))
        n_val = int(np.ceil(num_actors*val_perc))
        n_train = int(num_actors - (n_val + n_test))

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
'''
def build_matrix_dataset(merged_predictors, actors_list):

    #load preprocessing dict and output numpy matrices of predictors and target
    #containing only samples defined in actors_list

    predictors = np.array([])
    target = np.array([])
    index = 0
    total = len(actors_list)
    for i in actors_list:
        if i == actors_list[0]:  #if is first item
            predictors = np.array(merged_predictors[i])
            predictors = np.expand_dims(predictors, axis=0)
            #target = np.array(merged_target[i],dtype='float32')
            #print (i, predictors.shape)
        else:
            if np.array(merged_predictors[i]).shape != (0,):  #if it not void due to preprocessing errors
                expanded_predictors = np.expand_dims(merged_predictors[i], axis=0)
                predictors = np.concatenate((predictors, np.array(expanded_predictors)), axis=0)
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
'''


def build_matrix_dataset(merged_predictors, merged_target, actors_list):
    '''
    load preprocessing dict and output numpy matrices of predictors and target
    containing only samples defined in actors_list
    '''
    predictors = []
    target = []
    index = 0
    total = len(actors_list)
    for i in actors_list:
        for j in range(merged_predictors[i].shape[0]):
            predictors.append(merged_predictors[i][j])
            target.append(merged_target[i][j])
        index += 1
        perc = int(index / total * 20)
        perc_progress = int(np.round((float(index)/total) * 100))
        inv_perc = int(20 - perc - 1)
        string = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
        print ('\r', string, end='')
    predictors = np.array(predictors)
    target = np.array(target)
    print(' | shape: ' + str(predictors.shape))
    print ('\n')

    return predictors, target


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

def get_dataset_matrices(data_path, num_folds, num_fold, percs, train_path, val_path, test_path,
                        recompute_matrices = False):
    if recompute_matrices:
        #compute which actors put in train, val, test for current fold
        data_merged = np.load(data_path, allow_pickle=True)
        data_merged = data_merged.item()
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
        #TRAIN/VAL/TEST IN A BALANCED WAY
        foldable_list = list(data_merged.keys())
        fold_actors_list = folds_generator(num_folds, foldable_list, percs)
        train_list = fold_actors_list[int(num_fold)]['train']
        val_list = fold_actors_list[int(num_fold)]['val']
        test_list = fold_actors_list[int(num_fold)]['test']
        #del dummy
        print ('\n building dataset for current fold')
        print ('\n training:')
        training_data = build_matrix_dataset(data_merged, train_list)
        print ('\n validation:')
        validation_data = build_matrix_dataset(data_merged, val_list)
        print ('\n test:')
        test_data = build_matrix_dataset(data_merged, test_list)

        np.save(train_path, training_data)
        np.save(val_path, validation_data)
        np.save(test_path, test_data)

    else:
        if not os.path.exists(test_path):
            #load merged dataset, compute and save current tensors

            #compute which actors put in train, val, test for current fold
            data_merged = np.load(data_path, allow_pickle=True)
            data_merged = data_merged.item()
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
            #TRAIN/VAL/TEST IN A BALANCED WAY
            foldable_list = list(data_merged.keys())
            fold_actors_list = folds_generator(num_folds, foldable_list, percs)
            train_list = fold_actors_list[int(num_fold)]['train']
            val_list = fold_actors_list[int(num_fold)]['val']
            test_list = fold_actors_list[int(num_fold)]['test']
            #del dummy

            print ('\n building dataset for current fold')
            print ('\n training:')
            training_data = build_matrix_dataset(data_merged, train_list)
            print ('\n validation:')
            validation_data = build_matrix_dataset(data_merged, val_list)
            print ('\n test:')
            test_data = build_matrix_dataset(data_merged, test_list)

            np.save(train_path, training_data)
            np.save(val_path, validation_data)
            np.save(test_path, test_data)

        else:
            #load pre-computed tensors
            training_data = np.load(train_path)
            validation_data = np.load(val_path)
            test_data = np.load(test_path)

    return training_data, validation_data, test_data



def save_data(dataloader, model, device,epoch, gen_figs_path, gen_sounds_path, save_figs, save_sounds,
                save_items_epochs, save_items_n, features_type, dataset, warm_value_reparametrize,
                gen_distributions_path, save_latent_distribution):
    data_gen = []
    data_truth = []
    latent_dims = []
    labels = []
    if save_figs or save_sounds:
        if epoch % save_items_epochs == 0: #save only every n epochs
            #create folders
            if save_figs:
                curr_figs_path = os.path.join(gen_figs_path, dataset , 'epoch_'+str(epoch))
                if not os.path.exists(curr_figs_path):
                    os.makedirs(curr_figs_path)
            if save_sounds:
                curr_sounds_path = os.path.join(gen_sounds_path, dataset , 'epoch_'+str(epoch))
                if not os.path.exists(curr_sounds_path):
                    os.makedirs(curr_sounds_path)
            if save_latent_distribution:
                curr_distribution_path = os.path.join(gen_distributions_path, dataset , 'epoch_'+str(epoch))
                if not os.path.exists(curr_distribution_path):
                    os.makedirs(curr_distribution_path)

            for i, (sounds, truth) in enumerate(dataloader):
                sounds = sounds.to(device)
                target = sounds.cpu().numpy()
                batch_labels = truth.cpu().numpy()
                #compute predictions
                outputs, mu, logvar = model(sounds, warm_value_reparametrize)
                outputs = outputs.cpu().numpy()
                latents = mu.cpu().numpy()
                #concatenate predictions
                for single_vec in latents:
                    latent_dims.append(single_vec.reshape(np.max(single_vec.shape)))
                for label in batch_labels:
                    labels.append(label)
                for single_sound in outputs:
                    if features_type == 'waveform':
                        data_gen.append(single_sound)
                    elif features_type == 'spectrum':
                        data_gen.append(single_sound.reshape(single_sound.shape[-2], single_sound.shape[-1]))
                for single_sound in target:
                    if features_type == 'waveform':
                        data_truth.append(single_sound)
                    elif features_type == 'spectrum':
                        data_truth.append(single_sound.reshape(single_sound.shape[-2], single_sound.shape[-1]))

            #save items
            for i in range(save_items_n):
                if save_figs:
                    fig_name = 'gen_' + str(i) + '.png'
                    fig_path = os.path.join(curr_figs_path, fig_name)
                    plt.subplot(211)
                    plt.pcolormesh(data_gen[i].T)
                    plt.title('gen')
                    plt.subplot(212)
                    plt.pcolormesh(data_truth[i].T)
                    plt.title('original')
                    plt.savefig(fig_path)
                    plt.close()
                if save_sounds:
                    #generated
                    sound_name = 'gen_' + str(i) + '.wav'
                    sound_path = os.path.join(curr_sounds_path, sound_name)
                    sound = data_gen[i]
                    sound = sound.flatten()
                    sound = np.divide(sound, np.max(sound))
                    sound = np.multiply(sound, 0.8)
                    wavwrite(sound, SR, sound_path)
                    #originals
                    orig_name = 'orig_' + str(i) + '.wav'
                    orig_path = os.path.join(curr_sounds_path, orig_name)
                    orig = data_truth[i]
                    orig = orig.flatten()
                    orig = np.divide(orig, np.max(orig))
                    orig = np.multiply(orig, 0.8)
                    wavwrite(orig, SR, orig_path)

            if save_latent_distribution:
                fig_distribution_name = 'hidden_distribution.png'
                fig_distribution_path = os.path.join(curr_distribution_path, fig_distribution_name)
                z_embedded = TSNE(n_components=2).fit_transform(latent_dims)
                plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels,
                            alpha=.4, s=3**2, cmap='plasma')
                plt.colorbar()

                plt.savefig(fig_distribution_path)
                plt.close()

            print ('')
            if save_figs:
                print ('Generated figures saved')
            if save_sounds:
                print ('Generated sounds saved')

def plot_train_dict(dict_path):
    dict = np.load(dict_path, allow_pickle=True)
    dict = dict.item()
    keys = list(dict.keys())
    keys = list(filter(lambda x: 'train' in x, keys))
    legend = []
    for key in keys:
        if 'culo' not in key:
            legend.append(key)
            plt.plot(dict[key])
    plt.legend(legend)
    plt.show()
