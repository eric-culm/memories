from __future__ import print_function
import loadconfig
import configparser
import utility_functions as uf
import random
import preprocessing_utils as pre
import numpy as np
import librosa
import os, sys
import math

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file


#get values from config file
DUR = cfg.getint('main', 'dur')
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))



INPUT_FOLDER = sys.argv[1]
DATASET_NAME = sys.argv[2]
DATA_N = sys.argv[3]
#FEATURES_TYPE = sys.argv[3]

def get_dummy(filename):
    return 0.

def get_label_sc09(filename):

    if 'Zero' in filename:
        label = 0
    elif 'One' in filename:
        label = 1
    elif 'Two' in filename:
        label = 2
    elif 'Three' in filename:
        label = 3
    elif 'Four' in filename:
        label = 4
    elif 'Five' in filename:
        label = 5
    elif 'Six' in filename:
        label = 6
    elif 'Seven' in filename:
        label = 7
    elif 'Eight' in filename:
        label = 8
    elif 'Nine' in filename:
        label = 9

    return label

def get_label_nsynth(filename):

    if 'bass_electronic' in filename:
        label = 0
    elif 'bass_synthetic' in filename:
        label = 1
    elif 'keyboard_acoustic' in filename:
        label = 2
    elif 'keyboard_electronic' in filename:
        label = 3
    elif 'keyboard_synthetic' in filename:
        label = 4
    elif 'mallet_acoustic' in filename:
        label = 5
    elif 'brass_acoustic' in filename:
        label = 6
    elif 'organ_electronic' in filename:
        label = 7
    elif 'flute_acoustic' in filename:
        label = 8
    elif 'flute_synthetic' in filename:
        label = 9
    elif 'guitar_acoustic' in filename:
        label = 10
    elif 'guitar_electronic' in filename:
        label = 11
    elif 'reed_acoustic' in filename:
        label = 12
    elif 'string_acoustic' in filename:
        label = 13
    elif 'vocal_acoustic' in filename:
        label = 14
    elif 'vocal_synthetic' in filename:
        label = 15
    else:
        label = 16

    return label



def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    sounds_list = os.listdir(INPUT_FOLDER)  #get list of all soundfile paths
    sounds_list = list(filter(lambda x: x[-3:] == "wav", sounds_list))  #get only wav
    sounds_list = [os.path.join(INPUT_FOLDER, x) for x in sounds_list]  #append full path
    random.shuffle(sounds_list)

    if DATA_N != 'all':
        sounds_list = sounds_list[:int(DATA_N)]

    num_files = len(sounds_list)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + appendix + '_target.npy')
    index = 1  #index for progress bar
    for i in sounds_list:
        #print progress bar
        uf.print_bar(index, num_files)
        curr_list = [i]
        #compute predictors and target
        if 'sc09' in DATASET_NAME:
            get_label_func = get_label_sc09
        if 'nsynth' in DATASET_NAME:
            get_label_func = get_label_nsynth
        if 'chorder' in DATASET_NAME:
            get_label_func = get_dummy
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, DUR, get_label_func, True)
        print (curr_predictors.shape)
        #append preprocessed predictors and target to the dict
        if not np.isnan(np.std(curr_predictors)):  #cut empty sounds
            predictors[i] = curr_predictors
            target[i] = curr_target

        index +=1

    if len(list(predictors.keys())) == 1:
        data = predictors[list(predictors.keys())[0]]
        targ = target[list(predictors.keys())[0]]
        num_data = data.shape[0]
        predictors = {}
        target = {}
        for i in range(num_data):
            predictors[i] = data[i]
            target[i] = targ[i]


    #save dicts
    print ('\nSaving data...')
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)
    #print dimensions
    count = 0
    predictors_dims = 0
    keys = list(predictors.keys())
    for i in keys:
        count += predictors[i].shape[0]
    pred_shape = np.array(predictors[keys[0]]).shape[1:]
    tg_shape = np.array(target[keys[0]]).shape[1:]
    print ('')
    print ('MATRICES SUCCESFULLY COMPUTED')
    print ('')
    print ('Total number of datapoints: ' + str(count))
    print (' Predictors shape: ' + str(pred_shape))
    print (' Target shape: ' + str(tg_shape))


if __name__ == '__main__':
    main()
