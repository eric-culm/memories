import loadconfig
import configparser
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from audtorch import metrics
from scipy.stats import pearsonr
from scipy.signal import hilbert, resample
from scipy.fftpack import fft
from threading import Thread
import sounddevice as sd
from multiprocessing import Process
import time

import numpy as np
import utility_functions as uf
import losses

memory_bag = np.load('/home/eric/Desktop/memories/dataset/matrices/digits_validation_predictors_fold_0.npy')
#memory_bag = torch.tensor(memory_bag)
input = memory_bag[29]
#memory_bag = [input,input]
'''
culo = fft(input)
#culo = np.linspace(0.0, 1.0/(2.0*32000), 32000//2)
culo = np.abs(culo[0:len(culo)//4])
culo2 = np.abs(hilbert(culo))
culo2 = resample(culo2, 100)

plt.subplot(212)
plt.plot(culo)
plt.subplot(211)
plt.plot(culo2)
'''


plt.show()
class FilterStream:
    '''
    Compare amplitude and spectral envelopes of one input sound to all sounds present in the
    memory bag. If the input sound is enough similar, it passes through, else is discarded.
    '''
    def __init__(self, memory_bag, threshold, random_prob, env_length=100):
        self.memory_bag = memory_bag
        self.memory_bag_env = []
        self.memory_bag_spenv = []
        self.threshold = threshold
        self.random_prob = random_index
        self.env_length = env_length  #low = higher similarities: downsamples envelopes
        #compute envelope of memory_bag sounds
        #compute amp envelope
        #env = resample(np.abs(hilbert(i)), self.env_length)
        #compute spectral envelope
        #spenv = resample(np.abs(fft(i)[0:len(i)//4]), self.env_length)
        for i in self.memory_bag:
            self.memory_bag_env.append(resample(np.abs(hilbert(i)), self.env_length))
            self.memory_bag_spenv.append(resample(np.abs(fft(i)[0:len(i)//4]), self.env_length))

    def get_similarity_env(self, in_sound):
        #compute similarity between amplitude envelopes of input_sound
        #with all memory_bag sounds
        #RETURN THE FIRST SIMILARITY ABOVE THRESHOLD
        output = 0
        in_env = resample(np.abs(hilbert(in_sound)), self.env_length)
        for ref_env in self.memory_bag_env:
            similarity, p = pearsonr(ref_env, in_env)
            if similarity >= self.threshold:
                #print (similarity)
                break
        return similarity


    def get_similarity_spenv(self, in_sound):
        #compute similarity between spectral envelopes of input_sound
        #with all memory_bag sounds
        #RETURN THE FIRST SIMILARITY ABOVE THRESHOLD
        output = 0
        in_spenv = resample(np.abs(hilbert(in_sound)), self.env_length)
        for ref_spenv in self.memory_bag_spenv:
            similarity, p = pearsonr(ref_spenv, in_spenv)
            if similarity >= self.threshold:
                #print (similarity)
                break
        return similarity

    def filter_sound(self, in_sound):
        amp_similarity = self.get_similarity_env(in_sound)
        sp_similarity = self.get_similarity_spenv(in_sound)
        #if amp OR spectral similarity is above thresh
        if amp_similarity + sp_similarity >= self.threshold:
            output = in_sound
        else:
            #or if randomly chosen even if not similar
            random_prob = np.random.rand()
            if random_prob >= self.random_prob:
                output = in_sound
            else:
                #if none of the above
                output = None

        return output

class RecLoop(Thread):
    def __init__(self, dur, channel):
        Thread.__init__(self)
        self.dur = dur
        self.channel = channel
        self.buffer = np.zeros(dur)

    def rec_callback(self, indata, outdata, frames, time, status):
        '''
        record circular buffers of length DUR, updating every bloch size
        '''

        if status:
            print(status)
        self.buffer = np.roll(self.buffer, -512, axis=0)  #shift vector
        self.buffer[-frames:] = indata[:self.channel]  #add new data


    def rec(self):
        with sd.Stream(channels=self.channel, blocksize=512, callback=self.rec_callback):
            print ("")
            print ("Audio stream started")  #audio recording stream
            input()

    def meter(self):

        peak = max(abs(self.buffer))
        max_range = 10
        peak = int((peak**0.3)*max_range)
        print (peak)
        print ("METER: " + "|" + "|""="*peak*3 + "=D")


    def play(self):
        sd.play(self.buffer, 16000)



'''
a = input.copy()
r = sd.rec(44100, 44100, 1, mapping=2)
sd.play(r, 44100)
'''
del input

rec = RecLoop(32000,1)
rec.rec()

while True:
    rec.meter()
    time.sleep(0.2)
