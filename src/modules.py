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
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import utility_functions as uf
import losses
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)


class InputChannel:
    '''
    record a sliding buffer from a desired input channel
    '''
    def __init__(self, dur, channel, total_channels):
        self.dur = dur
        self.channel = channel
        self.total_channels = total_channels
        self.buffer = np.zeros(dur)
        self.stream =  sd.InputStream(channels=self.total_channels,
                                    blocksize=512 , callback=self.rec_callback)

    def rec_callback(self, indata, frames, time, status):
        '''
        record sliding buffers of length DUR, updating every bloch size
        '''
        if status:
            print(status)
        self.buffer = np.roll(self.buffer, -frames , axis=0)  #shift vector
        self.buffer[-frames:] = indata[:,self.channel] #add new data

    def rec(self, flag):
        if flag == 1:
            print ("")
            print ("Input stream started on channel: " + str(self.channel))
            print ("")
            self.stream.start()
        if flag == 0:
            print ("")
            print ("Input stream closed on channel: " + str(self.channel))
            print ("")
            self.stream.stop()

    def get_buffer(self):
        return self.buffer

    def meter_continuous(self, flag):
        self.meterflag = flag
        while self.meterflag == 1:
            peak = max(abs(self.buffer))
            print_peak = str(np.round(peak, decimals=3))
            meter_string =  "IN " + str(self.channel) + ": " + print_peak
            #print ('\r', meter_string, end='')
            print(meter_string)
            time.sleep(0.05)

    def meter_instantaneous(self):
        peak = max(abs(self.buffer))
        print_peak = str(np.round(peak, decimals=3))
        meter_string =  "IN " + str(self.channel) + ": " + print_peak
        #print ('\r', meter_string, end='')
        print (meter_string)


class FilterSound:
    '''
    Compare amplitude and spectral envelopes of one input sound to all sounds present in the
    memory bag. If the input sound is enough similar, it passes through, else is discarded.
    '''
    def __init__(self, memory_bag, threshold, random_prob, env_length=100):
        self.memory_bag = memory_bag
        self.memory_bag_env = []
        self.memory_bag_spenv = []
        self.threshold = threshold
        self.random_prob = random_prob
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
        mean_similarity = (amp_similarity + sp_similarity) / 2
        #if amp OR spectral similarity is above thresh
        if mean_similarity >= self.threshold:
            output = in_sound
        else:
            #or if randomly chosen even if not similar
            random_prob = np.random.rand()
            if random_prob <= self.random_prob:
                output = in_sound
            else:
                #if none of the above
                output = None

        return output, mean_similarity

class FilterStream:
    '''
    Apply FilterSound to multiple input sounds and collects a bag
    of filtered sounds
    '''
    def __init__(self, frequency, streaming_object, filtering_object):
        self.frequency = frequency
        self.flag = 0
        self.streaming_object = streaming_object
        self.filtering_object = filtering_object
        self.bag = []

    def filter_input_sound(self):
        buffer = self.streaming_object.get_buffer()
        filtered, sim = self.filtering_object.filter_sound(buffer)
        return filtered

    def filter_stream(self, flag):
        self.flag = flag
        if self.flag == 1:
            self.bag = []
        while self.flag == 1:
            filtered = self.filter_input_sound()
            if filtered is not None:
                self.bag.append(filtered)
            time.sleep(self.frequency)

    def get_bag(self):
        return self.bag

class DummyModel(nn.Model):
    def __init__(self, dur, latent_dim):
        super(DummyModel, self).__init__()
        self.dur = dur
        self.latent_dim = latent_dim

    def encode(self, x):
        return torch.sigmoid(torch.randn(self.latent_dim))

    def decode(self, z):
        return torch.tanh(torch.randn(self.dur))


class LatentOperators:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def random_slice(self):
        random_perc = np.random.randint(self.latent_dim)
        start = np.random.randint(self.latent_dim - random_perc)
        end = start + random_perc
        return [start, end]

    def add(self, x1, x2):
        return torch.add(x1,x2)

    def sub(self, x1, x2):
        return torch.sub(x1,x2)

    def mul(self, x1,x2):
        return torch.mul(x1,x2)

    def div(self, x1,x2):
        return torch.mul(x1,x2)
