import numpy as np
import utility_functions as uf
import postprocessing_utils as pp
import matplotlib.pyplot as plt
import librosa
import scipy
from modules import *

post = Postprocessing(16000, '../IRs/revs/divided/')

s1 = '/home/eric/Downloads/voice_prova.wav'


out_file = '/home/eric/Desktop/quick_sounds/CULISMO'
dur = 16000
sounds = []
samples, sr = librosa.core.load(s1)
seg = np.arange(0, len(samples), dur)

for i in seg:
    try:
        sounds.append(samples[i:i+dur])
    except:
        pass

def splitter(x, sr, min_len=250):
    uf.wavwrite(x, sr, out_file+'0.wav')
    threshold = 4.
    min_len = sr*min_len /1000
    onsets = []
    onset_env = librosa.onset.onset_strength(x, sr)
    for i in range(onset_env.shape[0]):
        if onset_env[i] > threshold:
            onsets.append(i)
    step = x.shape[0] / onset_env.shape[0]
    onsets = np.multiply(onsets, step)
    filtered_onsets = [0]

    for i in range(onsets.shape[0]-1):
        if i == 0:
            if onsets[i] >= min_len:
                filtered_onsets.append(int(onsets[i]))

        else:
            if onsets[i] - onsets[i-1] >= min_len or onsets[i] - filtered_onsets[-1 ] >= min_len:
                filtered_onsets.append(int(onsets[i]))

    if x.shape[0] - filtered_onsets[-1] < min_len:
        del filtered_onsets[-1]

    output = []
    filtered_onsets = np.array(filtered_onsets)
    if filtered_onsets.shape[0] > 1:
        for i in range(filtered_onsets.shape[0]-1):
            output.append(x[filtered_onsets[i]:filtered_onsets[i+1]])
        output.append(x[filtered_onsets[-1]:])

    '''
    index = 1
    for i in output:
        print (i.shape)
        uf.wavwrite(i, sr, out_file+str(index)+'.wav')
        index+=1
    '''
    return output

def xfade(x1, x2, ramp):
    out = []
    fadein = np.arange(ramp) / ramp
    fadeout = np.arange(ramp, 0, -1) / ramp

    x1[-ramp:] = x1[-ramp:] * fadeout
    x2[:ramp] = x2[:ramp] * fadein

    left = x1[:-ramp]
    center = x1[-ramp:] + x2[:ramp]
    end = x2[ramp:]

    return np.concatenate((left,center,end), axis=0)


def concat_split s(sounds, sr, fade_len, min_len, max_len):
    max_len_samps = sr * max_len
    all_splits = []
    index = 1
    print ('analyzing sounds')
    for i in sounds:
        a = splitter(i, 16000)
        for j in a:
            all_splits.append(j)
        uf.print_bar(index, len(sounds))
        index+=1
    len_out = 0
    output = list(all_splits[np.random.randint(len(sounds))])
    while len_out < max_len_samps:
        random_i = np.random.randint(len(sounds))
        curr_sound = all_splits[random_i]
        if len(curr_sound) > min_len:
            output = xfade(output, all_splits[random_i], fade_len)
        len_out = len(output)
    output = np.array(output)
    output = post.reverb(output, 'any')
    uf.wavwrite(output, sr, out_file+'aaa.wav')
    print ('\nsound built')

concat_splits(sounds, sr, 20, 20)
