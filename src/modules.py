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
from audtorch import metrics
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
import pyrubberband as rub
import librosa
import random
import scipy
import os

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
SR = cfg.getint('sampling', 'sr_target')
CLIENT_IP = cfg.get('osc', 'client_ip')
GRID_LT_PATH = cfg.get('main', 'grid_lt_path')
GRID_ST_PATH = cfg.get('main', 'grid_st_path')


class Memory:
    '''
    long-term and short-term memory bags of the system
    '''
    def __init__(self, memory_lt_path, memory_st_path, memory_st_limit=100, memory_rt_limit=100):
        self.memory_st_limit = memory_st_limit
        self.memory_rt_limit = memory_rt_limit
        self.memory_st_path = memory_st_path
        self.memory_lt = list(np.load(memory_lt_path, allow_pickle=True))
        self.memory_rt = []
        try:
            self.memory_st = list(np.load(memory_st_path, allow_pickle=True))
        except:
            self.memory_st = []

    def get_state(self):
        return len(self.memory_lt), len(self.memory_st), len(self.memory_rt)

    def get_memory_lt(self):
        return self.memory_lt

    def get_memory_st(self):
        return self.memory_st

    def get_memory_rt(self):
        return self.memory_rt

    def save_memory_st(self):
        np.save(self.memory_st_path, self.memory_st)

    def append_to_st(self, input_list):
        if not isinstance(input_list, list):
            input_list = [input_list]
        for i in input_list:
            if len(self.memory_st) >= self.memory_st_limit:
                del self.memory_st[0]
            self.memory_st.append(i)

    def append_to_rt(self, input_list):
        if not isinstance(input_list, list):
            input_list = [input_list]
        for i in input_list:
            if len(self.memory_rt) >= self.memory_rt_limit:
                del self.memory_rt[0]
            self.memory_rt.append(i)

    def del_st(self):
        self.memory_st = []

    def del_rt(self):
        self.memory_rt = []


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
        print ('Transfering to client')
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
    def __init__(self, dur, channel, total_channels, sr):
        self.dur = dur
        self.channel = channel
        self.total_channels = total_channels
        self.buffer = np.zeros(dur)
        self.stream =  sd.InputStream(channels=self.total_channels, samplerate=sr,
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

    def filter_stream(self, flag, channel, memory, memory_type):
        self.flag = flag
        if self.flag == 1:
            print ("\nStarted storing stimuli from channel " + str(channel)) + ' to memory ' + memory_type
            self.bag = []
        else:
            print ("\nStopped storing stimuli from channel " + str(channel))+ ' to memory ' + memory_type
        while self.flag == 1:
            filtered = self.filter_input_sound()
            if filtered is not None:
                self.bag.append(filtered)
                if memory_type == 'st':
                    memory.append_to_st(filtered)
                if memory_type == 'rt':
                    memory.append_to_rt(filtered)
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

    def blur(self, x, mul):
        noise = (torch.randn(self.latent_dim) * mul).reshape(1, self.latent_dim)
        return torch.add(x, noise)

    def spike(self, x, n_peaks):
        noise = np.zeros(self.latent_dim)
        for i in range(n_peaks):
            pos = np.random.randint(self.latent_dim)
            peak = np.random.randint(self.latent_dim) / float(self.latent_dim)
            noise[pos] = peak * 0.3
        noise = torch.tensor(noise).float().reshape(1, self.latent_dim)

        return torch.add(x, noise)


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
        self.model.eval()
        self.dim = 16384
        try:
            self.grid_lt = list(np.load(GRID_LT_PATH, allow_pickle=True))
        except:
            self.grid_lt = []
        try:
            self.grid_st = list(np.load(GRID_ST_PATH, allow_pickle=True))
        except:
            self.grid_st = []
        self.CCC =  metrics.ConcordanceCC()


    def encode_ext(self, x):
        x = torch.tensor(x).float().reshape(1,1,self.dim)
        mu, logvar = self.model.enc_func(x)
        return mu

    def decode_ext(self, z):
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        x = x / np.max(x)
        x = x * 0.8
        return x

    def encode_int(self, x):
        #x = torch.tensor(x).float().reshape(1,1,self.dim)
        mu, logvar = self.model.enc_func(x)
        return mu

    def decode_int(self, z):
        #z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        x = x / np.max(x)
        x = x * 0.8
        return x

    def get_z_loss(self, x):
        x = torch.tensor(x).float().reshape(1,1,self.dim)
        mu, logvar = self.model.enc_func(x)
        recon_x = self.model.dec_func(mu).reshape(self.dim)
        recon_loss = 1 - torch.abs(self.CCC(recon_x, torch.squeeze(x))).detach().numpy()
        return mu, recon_loss

    def gen_random_x(self):
        max_grid = np.max(np.array(self.grid_lt))
        z = np.random.rand(self.model.latent_dim) * 0.5 + 0.5
        z = z / max_grid
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        return x

    def gen_random_z(self):
        max_grid = np.max(np.array(self.grid_lt))
        z = np.random.rand(self.model.latent_dim) * 0.5 + 0.5
        z = z * max_grid
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)

        return z

    def gen_random_peak(self):
        max_grid = np.max(np.array(self.grid_lt))
        z = np.zeros(self.model.latent_dim)
        pos = np.random.randint(self.model.latent_dim)
        peak = np.random.randint(self.model.latent_dim) / float(self.model.latent_dim)
        peak = peak / max_grid
        z[pos] = peak
        z = torch.tensor(z).float().reshape(1,self.model.latent_dim)
        x = self.model.dec_func(z).reshape(self.dim).detach().numpy()
        return x

    def quantize(self, z, grid_type='lt'):
        if grid_type == 'lt':
            grid = torch.tensor(self.grid_lt)
        if grid_type == 'st':
            grid = torch.tensor(self.grid_st)
        dists = []
        for i in grid:
            curr_dist = self.CCC(z, i)
            dists.append(curr_dist.cpu().numpy())
        nearest_z = np.argmax(dists)
        return grid[nearest_z]


    def compute_quantization_grid(self, memory, memory_type, threshold=0.3):
        print('\nComputing quantization grid')
        temp_list = []
        if not isinstance(memory, list):
            input_list = [memory]
        index=1
        for sound in memory:
            z, loss = self.get_z_loss(sound)
            temp_list.append((z.detach().numpy(), loss))
            uf.print_bar(index, len(memory))
            index += 1
        sorted_list = sorted(temp_list, key=lambda x: x[1])  #sort from lowest to highest CCC
        filtered_list = list(filter(lambda x: x[1] <= threshold, sorted_list))  #cut items with highest CCC
        filtered_list = [x[0] for x in filtered_list]  #cut CCCs from list
        if memory_type == 'lt':
            self.grid_lt = filtered_list
            np.save(GRID_LT_PATH, filtered_list)
        if memory_type == 'st':
            self.grid_st = filtered_list
            np.save(GRID_ST_PATH, filtered_list)
        print ('\nGrid ' + str(memory_type) + ' succesfully saved')
        print ('Num datapoints: ' + str(len(filtered_list)) + ' | Discarded ' + str(len(sorted_list) - len(filtered_list)))

class Postprocessing:
    def __init__(self, sr, irs_path):
        self.sr = sr
        self.irs_path = irs_path
        self.eps = 0.0001


    def strip_silence(self, samples, threshold=35):
        #strip initial and final silence
        cut = pp.strip_silence(samples, threshold)
        return cut


    def gen_silence(self, dur):
        dur_samps = int(dur * self.sr)
        noise = (np.random.sample(dur_samps) * 2) - 1
        noise = noise * self.eps
        return noise

    def paulstretch(self, samples, stretch_factor, winsize, transients_level):
        #extreme time stretching with crazy algo
        print ('paulstretching:')
        stretched = pp.paulstretch_wrap(self.sr, samples, stretch_factor, winsize, transients_level)
        return stretched

    def stretch(self, samples, stretch_factor, granularity=5):
        #phase vocoder based stretching
        stretched = rub.pyrb.time_stretch(samples, self.sr, stretch_factor, {'-c':granularity})
        return stretched

    def reverb(self, samples, rev_type):
        #convolution with randomly-selected impulse response
        if rev_type == 'any':
            contents = os.listdir(self.irs_path)
            rev_type_sel = random.choice(contents)
        else:
            rev_type_sel = rev_type

        folder = os.path.join(self.irs_path, rev_type_sel)
        irs = os.listdir(folder)
        random_rev = random.choice(irs)
        random_rev = os.path.join(folder, random_rev)
        rev_amount = np.random.rand() * 0.7 + 0.3

        IR, sr = librosa.core.load(random_rev, mono=False, sr=self.sr)

        try:  #convert to mono if sample is stereo
            IR = IR[0]
        except:
            pass

        IR = IR * rev_amount

        out = scipy.signal.convolve(samples, IR)
        out = out / max(out)
        out = out * 0.9
        return out

    def convolve(self, samples1, samples2):
        #convolve 2 signals
        out = scipy.signal.convolve(samples1, samples2)
        out = out / max(out)
        out = out * 0.9
        return out


    def splitter(self, x, min_len=250):
        #segment sound according to changes in the spectral shape
        #uf.wavwrite(x, sr, out_file+'0.wav')

        threshold = 4.
        min_len = self.sr*min_len /1000
        onsets = []
        onset_env = librosa.onset.onset_strength(x, self.sr)
        for i in range(onset_env.shape[0]):
            if onset_env[i] > threshold:
                onsets.append(i)
        step = x.shape[0] / onset_env.shape[0]
        onsets = np.multiply(onsets, step)
        filtered_onsets = [0]
        '''
        print ('\nfiga')
        print (filtered_onsets)
        '''

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
        else:
            output = [x]

        '''
        index = 1
        for i in output:
            print (i.shape)
            uf.wavwrite(i, sr, out_file+str(index)+'.wav')
            index+=1
        '''

        return output

    def xfade(self, x1, x2, ramp):
        #simple linear crossfade and concatenation
        out = []
        fadein = np.arange(ramp) / ramp
        fadeout = np.arange(ramp, 0, -1) / ramp

        x1[-ramp:] = x1[-ramp:] * fadeout
        x2[:ramp] = x2[:ramp] * fadein

        left = x1[:-ramp]
        center = x1[-ramp:] + x2[:ramp]
        end = x2[ramp:]

        return np.concatenate((left,center,end), axis=0)


    def concat_split_old(self, sounds, out_len, perc_stretched, stretch_f_center=0.7,
                    stretch_f_excursion=0.2, fade_len=30, min_len=250):
        #create long file concatenation splits
        out_len_samps = self.sr * out_len
        all_splits = []
        index = 1
        print ('analyzing sounds')
        for i in sounds:
            a = self.splitter(i, 16000)
            for j in a:
                all_splits.append(j)
            uf.print_bar(index, len(sounds))
            index+=1
        len_out = 0

        output = list(all_splits[np.random.randint(len(sounds))])
        while len_out < out_len_samps:
            random_i = np.random.randint(len(sounds))
            stretch_flag = np.random.sample() < perc_stretched
            curr_sound = all_splits[random_i]

            if stretch_flag:
                stretch_factor = (np.random.sample() * stretch_f_excursion * 2) + selp.eps
                stretch_factor = stretch_factor + (stretch_f_center - stretch_f_excursion)
                print ('\nculo')
                print (stretch_factor)
                curr_sound = self.stretch(curr_sound, stretch_factor)

            if len(curr_sound) > min_len:
                output = self.xfade(output, all_splits[random_i], fade_len)
            len_out = len(output)
        output = np.array(output)

        return output

    def distribute_pan_stereo(self, sounds, bounds=[0,1]):

        n_sounds = len(sounds)
        lens = []
        for i in sounds:
            lens.append(len(i))
        max_len = max(lens)
        #create ramp
        ramp = np.arange(n_sounds) / (n_sounds-1)
        ramp = np.sqrt(ramp)  #better for panning
        ramp = np.interp(ramp, (0., 1.), (bounds[0], bounds[1]))
        pans = []
        #create tuple of multipliers
        for i in ramp:
            pans.append((i, 1-i))
        #create output vector stereo
        out = np.array([[np.zeros(max_len)],[np.zeros(max_len)]])
        #append panned sounds
        index = 0
        for i in sounds:
            pad = np.zeros(max_len)
            pad[:len(i)] = i
            left = pad * pans[index][0]
            right = pad * pans[index][1]
            out[0] = np.add(out[0], left)
            out[1] = np.add(out[1], right)
            index += 1

        out = np.divide(out, np.max(out))
        out = np.multiply(out, 0.8)
        out = np.squeeze(out)

        return out

    def cluster_data(self, sounds, n_clusters=5):
        output = {}
        for i in range(n_clusters):
            output[i] = []
        feats = []
        for i in sounds:
            feats.append(np.abs(scipy.fftpack.fft(i, 128)[1:64]))
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(feats)
        clusters = clustering.labels_
        for i in range(len(clusters)):
            label = clusters[i]
            output[label].append(sounds[i])
            
        return output



    def concat_split(self, sounds, out_len, sil_perc_curve, sil_len_curve,
                            stretch_perc_curve, stretch_factor_curve, fade_len=30,
                            min_len=250., len_curves=100):
        #create long file concatenation splits

        sil_len_bounds = [0.1, 6]
        sil_perc_bounds = [0, 1]

        stretch_factor_bound = 8
        stretch_perc_bounds = [0, 1]

        gaussian_steps = 5
        gaussian_range = 0.2

        sil_len_curve = sil_len_curve ** 2
        sil_len_curve =(sil_len_curve * (sil_len_bounds[1]-sil_len_bounds[0])) + sil_len_bounds[0]
        sil_perc_curve = (sil_perc_curve * (sil_perc_bounds[1]-sil_perc_bounds[0])) + sil_perc_bounds[0]

        stretch_factor_curve = stretch_factor_curve * 2  #from 0-1 to 0-2
        #stretch_factor_curve = stretch_factor_curve ** np.sqrt(stretch_factor_bound)  #turn to exponential with ~bounds
        stretch_perc_curve = (stretch_perc_curve * (stretch_perc_bounds[1]-stretch_perc_bounds[0])) + stretch_perc_bounds[0]

        out_len_samps = int(self.sr * out_len)
        all_splits = []
        index = 1
        print ('analyzing sounds')
        for i in sounds:
            a = self.splitter(i, 16000)
            for j in a:
                all_splits.append(j)
            uf.print_bar(index, len(sounds))
            index+=1
        len_out = 0

        output = self.gen_silence(0.1) #init out buffer with silence

        #re-order list clustering if wanted HERE!!

        print ('\nbuilding output')
        while len_out < out_len_samps:
            curr_spot = int(np.round(len_out / out_len_samps * len_curves))  #where you are in the score
            curr_spot = np.clip(curr_spot, 0, len_curves-1)
            #random probs
            silence_flag = np.random.sample() < sil_perc_curve[curr_spot]
            stretch_flag = np.random.sample() < stretch_perc_curve[curr_spot]

            #gen silence
            if silence_flag:
                #compute silence
                silence_gaussian_std = sil_len_curve[curr_spot] * gaussian_range
                silence_dur = np.random.normal(sil_len_curve[curr_spot], silence_gaussian_std) #gen random silence duration in secs
                silence_dur = np.clip(silence_dur, 0.05, 10000)
                curr_sound = self.gen_silence(silence_dur)

            else:
                random_i = np.random.randint(len(sounds))  #take random sound
                curr_sound = all_splits[random_i]

                #not stretching silences
                if stretch_flag:
                    #compute std taking random step in the curve
                    #stretch_gaussian_step = np.random.randint(gaussian_steps*2+1) - gaussian_steps  #random step in the curve
                    #stretch_gaussian_step = np.clip(stretch_gaussian_step, 0, len_curves-1)  #clip step to curve bounds
                    #stretch_gaussian_std = stretch_factor_curve[int(np.round(stretch_gaussian_step))]  #take value of std in the curve
                    stretch_gaussian_std = np.abs(stretch_factor_curve[curr_spot]-1) * gaussian_range

                    stretch_factor = np.random.normal(stretch_factor_curve[curr_spot], stretch_gaussian_std)  #lognormal distribution
                    if stretch_factor == 0:
                        stretch_factor = self.eps
                    stretch_factor = np.abs(stretch_factor)
                    stretch_factor = np.clip(stretch_factor, stretch_factor_curve[curr_spot]-stretch_gaussian_std, stretch_factor_curve[curr_spot]+stretch_gaussian_std)
                    stretch_factor = stretch_factor ** np.sqrt(stretch_factor_bound)  #turn to exponential within ~bounds
                    stretch_factor = np.clip(stretch_factor, 1/stretch_factor_bound, stretch_factor_bound)
                    curr_sound = self.stretch(curr_sound, stretch_factor)

                uf.print_bar(curr_spot, len_curves)

            if len(curr_sound) > min_len:
                output = self.xfade(output, curr_sound, fade_len)
            len_out = len(output)
        output = np.array(output)

        return output
