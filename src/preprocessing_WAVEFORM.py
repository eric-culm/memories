from __future__ import print_function
import loadconfig
import configparser
import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import librosa
import os, sys
import math

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file

SR = cfg.getint('sampling', 'sr_target')
DUR = cfg.getfloat('preprocessing', 'sequence_length')
SEQUENCE_LENGTH = cfg.getfloat('preprocessing', 'sequence_length')
SEQUENCE_OVERLAP = cfg.getfloat('preprocessing', 'sequence_overlap')
#in
#INPUT_RAVDESS_FOLDER =  cfg.get('preprocessing', 'input_audio_folder_ravdess')
INPUT_FOLDER = sys.argv[1]
DATASET_NAME = sys.argv[2]
FEATURES_TYPE = sys.argv[3]

#out
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

SEGMENTATION = False
print ('Segmentation: ' + str(SEGMENTATION))

def prepare_sound(input_sound, features_type):
    '''
    generate predictors (stft) and target (valence sequence)
    of one sound file from the OMG dataset
    '''
    raw_samples, sr = librosa.core.load(input_sound, sr=SR)  #read audio
    dur_samps = int(np.round(SR * DUR))
    if SEGMENTATION:
        # if segment cut initial and final silence if present
        samples = uf.strip_silence(raw_samples)

    else:
        #if not, zero pad all sounds to the same length
        samples = np.zeros(dur_samps)
        samples[:len(raw_samples)] = raw_samples[:dur_samps]  #zero padding
    #normalize
    features = pre.extract_features(samples, features_type)

    return features

def segment_datapoint(vector_input):
    '''
    segment features and annotations of one long audio file
    into smaller matrices of length "sequence_length"
    and overlapped by "sequence_overlap"
    '''
    num_frames = len(vector_input)

    step = (SEQUENCE_LENGTH*SR)*(SEQUENCE_OVERLAP*SR) #segmentation overlap step
    pointer = np.arange(0, num_frames, step, dtype='int')  #initail positions of segments
    predictors = []
    #slice arrays and append datapoints to vectors
    if SEGMENTATION:
        for start in pointer:
            stop = int(start + SEQUENCE_LENGTH)
            #print start_annotation, stop_annotation, start_features, stop_features
            if stop <= num_frames:
                temp_predictors = vector_input[start:stop]

                predictors.append(temp_predictors)
            else:  #last datapoint has a different overlap
                temp_predictors = vector_input[-int(SEQUENCE_LENGTH):]
                predictors.append(temp_predictors)
    else:
        predictors.append(vector_input)
    predictors = np.array(predictors)

    return predictors

def preprocess_datapoint(input_sound, features_type):

    sound_file = os.path.join(INPUT_FOLDER , input_sound)  #get correspective sound
    predictors = prepare_sound(sound_file, features_type)  #compute features
    if SEGMENTATION:
        predictors = segment_datapoint(long_predictors)   #slice feature maps
    predictors = np.array(predictors)

    return predictors


def main(input_folder):
    contents = os.listdir(INPUT_FOLDER)
    contents = list(filter(lambda x: x[-3:] == "wav", contents))
    #contents = contents [:5]
    num_sounds = len(contents)
    predictors = {}
    predictors_save_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + '_predictors.npy')
    index = 1
    for i in contents:
        curr_predictors = preprocess_datapoint(i, FEATURES_TYPE)
        if not np.isnan(np.std(curr_predictors)):
            print (curr_predictors.shape)
            predictors[i] = curr_predictors
        uf.print_bar(index, num_sounds)
        index += 1
    np.save(predictors_save_path, predictors)
if __name__ == '__main__':
    main(INPUT_FOLDER)
