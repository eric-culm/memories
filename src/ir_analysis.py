import configparser
import loadconfig
import os
import librosa
import utility_functions as uf
import numpy as np

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
MAIN_SR = cfg.getint('main', 'main_sr')
IRS_PATH = cfg.get('main', 'irs_path')

ir_lengths = {0:[],1:[],2:[],3:[],4:[]}
lens = []

def strip_silence(input_vector, threshold=60):
    split_vec = librosa.effects.split(input_vector, top_db = threshold)
    onset = split_vec[0][0]
    offset = split_vec[-1][-1]
    cut = input_vector[onset:offset]

    return cut

contents = list(filter(lambda x: '.wav' in x,os.listdir(IRS_PATH)))
index = 0
print ('analysis:')
for i in contents:
    curr_path = os.path.abspath(os.path.join(IRS_PATH, i))
    curr_sound, dummy = librosa.core.load(curr_path, sr=MAIN_SR)
    cut = strip_silence(curr_sound)
    curr_len = len(cut) / MAIN_SR
    lens.append(curr_len)
    if curr_len <= 0.5:
        ir_lengths[0].append(curr_path)
    elif curr_len > 0.5 and curr_len <= 1.:
        ir_lengths[1].append(curr_path)
    elif curr_len > 1. and curr_len <= 1.5:
        ir_lengths[2].append(curr_path)
    elif curr_len > 1.5 and curr_len <= 2.:
        ir_lengths[3].append(curr_path)
    elif curr_len > 2.:
        ir_lengths[4].append(curr_path)
    uf.print_bar(index, len(contents))
    index += 1

analysis_file_path = os.path.join(IRS_PATH, 'ir_analysis.npy')
np.save(analysis_file_path, ir_lengths)
print (ir_lengths)


import matplotlib.pyplot as plt
import numpy as np
plt.title('REVERB LENGTHS')
plt.hist(lens, normed=True)
plt.show()
