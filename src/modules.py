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
import subprocess
from multiprocessing import Process
import define_models as choose_model
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import utility_functions as uf
import configparser
import loadconfig
import os

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
SR = cfg.getint('sampling', 'sr_target')
CLIENT_IP = cfg.get('osc', 'client_ip')

class Memory:
    '''
    long-term and short-term memory bags of the system
    '''
    def __init__(self, memory_lt_path, memory_st_path, memory_st_limit=100):
        self.memory_st_limit = memory_st_limit
        self.memory_st_path = memory_st_path
        self.memory_lt = list(np.load(memory_lt_path))
        try:
            self.memory_st = list(np.load(memory_st_path))
        except:
            self.memory_st = []

    def get_state(self):
        return len(self.memory_lt), len(self.memory_st)

    def get_memory_lt(self):
        return self.memory_lt

    def get_memory_st(self):
        return self.memory_st

    def save_memory_st(self):
        np.save(self.memory_st_path, self.memory_st)

    def append_to_st(self, input_list):
        if not isinstance(input_list, list):
            input_list = [input_list]
        for i in input_list:
            if len(self.memory_st) >= self.memory_st_limit:
                del self.memory_st[0]
            self.memory_st.append(i)

    def del_st(self):
        self.memory_st = []


class Allocator:
    '''
    manage the shared folder between server and client.
    '''
    def __init__(self, server_shared_path, client_shared_path, sr=SR,
                client_ip=CLIENT_IP, client_username='eric'):
        self.sr = sr
        self.client_ip = client_ip
        self.client_username = client_username
        self.server_path = server_shared_path
        self.client_path = client_shared_path

    def write_local(self, input_list, query_name):
        print (np.array(input_list).shape)
        print ('Writing sounds to local shared path')
        output_path = os.path.join(self.server_path, query_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not isinstance(input_list, list):
            input_list = [input_list]
        index = 0
        for sound in input_list:
            curr_path = os.path.join(output_path, str(index)+'.wav')
            uf.wavwrite(sound, self.sr, curr_path)
            index += 1
        print ('All sounds written')

    def to_client(self, query_name):
        input_path = os.path.join(self.server_path, query_name)
        output_path = os.path.join(self.client_path, query_name)

        line = 'scp -pr ' + input_path + ' ' + self.client_username + '@' + self.client_ip + ':' + output_path
        process = subprocess.Popen(line, shell=True)
        process.communicate()
        process.wait()




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

    def filter_stream(self, flag, channel, memory):
        self.flag = flag
        if self.flag == 1:
            print ("\nStarted storing stimuli from channel: " + str(channel))
            self.bag = []
        else:
            print ("\nStopped storing stimuli from channel: " + str(channel))
        while self.flag == 1:
            filtered = self.filter_input_sound()
            if filtered is not None:
                self.bag.append(filtered)
                memory.append_to_st(filtered)
            time.sleep(self.frequency)


    def get_bag(self):
        return self.bag

class DummyModel(nn.Module):
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

    def __random_slice__(self):
        '''
        output bounds of a random slice within the latent dim
        '''
        random_perc = np.random.randint(self.latent_dim)
        random_invperc = np.random.randint(self.latent_dim)

        start_init = np.random.randint(self.latent_dim - random_perc)
        end_init = start_init + random_perc
        #randomly invert direction
        if random_invperc >= self.latent_dim/2:
            start = self.latent_dim - end_init
            end = self.latent_dim - start_init
        else:
            start = start_init
            end = end_init
        return [start, end]


    def __slice_wrapper__(self, func_name, x1, x2):
        '''
        apply func_name to x1, x2 only in a random slice
        '''
        func = getattr(self,func_name)
        start, end = self.__random_slice__()
        slice1 = x1[start:end]
        slice2 = x2[start:end]
        processed_slice = func(slice1, slice2)
        output = x1
        output[start:end] = processed_slice

        return output

    def add(self, x1, x2):
        return torch.sigmoid(torch.add(x1,x2))

    def sub(self, x1, x2):
        return torch.sigmoid(torch.sub(x1,x2))

    def mul(self, x1, x2):
        return torch.sigmoid(torch.mul(x1,x2))

    def div(self, x1, x2):
        return torch.sigmoid(torch.mul(x1,x2))

    def mean(self, x1, x2):
        concat = torch.stack((x1,x2))
        return torch.mean(concat,0)

    def xfade(self, x1, x2):
        ramp1 = np.arange(1, x1.shape[0]+1) / x1.shape[0]
        ramp2 = np.array(np.flip(ramp1))
        ramp1 = torch.tensor(ramp1).float()
        ramp2 = torch.tensor(ramp2).float()
        scaled1 = torch.mul(x1, ramp1)
        scaled2 = torch.mul(x2, ramp2)
        return torch.add(scaled1, scaled2)

class VAE_model:
    def __init__(self, architecture, weights_path, parameters, device):
        model_string = 'model_class, model_parameters = choose_model.' + architecture + '(1, 1, parameters)'
        exec(model_string)
        self.device = torch.device(device)
        weights = torch.load(weights_path,map_location=self.device)
        self.model = locals()['model_class'].to(self.device)
        self.model.load_state_dict(weights, strict=False)
        self.dim = 16384

    def encode_ext(self, x):
        x = torch.tensor(x).float().reshape(1,1,self.dim)
        mu, logvar = self.model.enc_func(x)
        return mu

    def decode_ext(self, z):
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        return x

    def gen_random(self):
        z = np.random.rand(self.model.latent_dim) * 0.5 + 0.5
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        return x

    def gen_random_peak(self):
        z = np.zeros(self.model.latent_dim)
        pos = np.random.randind(self.model.latent_dim)
        peak = np.random.randind(self.model.latent_dim) / float(self.model.latent_dim)
        z[pos] = peak
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        return x
