import torch
from torch import nn
from torch import optim
import warnings
import torch.nn.functional as F
import soundfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from audtorch import metrics
from scipy.stats import pearsonr
from scipy.signal import hilbert, resample
from scipy.fftpack import fft
from scipy import signal
from scipy.signal import iirfilter, lfilter, convolve
from threading import Thread
from audtorch import metrics
import sounddevice as sd
import subprocess
import copy
from multiprocessing import Process
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy as np
import utility_functions as uf
import configparser
import loadconfig
import pyrubberband as rub
import librosa
import random
import scipy
import threading
import os,sys,inspect
from srnn_models_map import *
import scene_constrains as sc

# insert at 1, 0 is the script path (or '' in REPL)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
import vae_pytorch.define_models as choose_model

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
SR = cfg.getint('sampling', 'sr_target')
MAIN_SR = cfg.getint('main', 'main_sr')
CLIENT_IP = cfg.get('osc', 'client_ip')
GRID_LT_PATH = cfg.get('main', 'grid_lt_path')
GRID_ST_PATH = cfg.get('main', 'grid_st_path')
SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')
IRS_PATH = cfg.get('main', 'irs_path')



class Memory:
    '''
    long-term short-term and real-time memory bags of the system
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
    def __init__(self, server_shared_path, client_shared_path, sr,
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
            swapped_sound = np.swapaxes(sound,0,1)
            soundfile.write(curr_path, swapped_sound, self.sr, format='WAV', subtype='PCM_24')
            #librosa.output.write_wav(curr_path, sound, self.sr)
            #uf.wavwrite(sound, self.sr, curr_path)
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
        return self.buffer.copy()

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

class Postprocessing:
    def __init__(self, sr=0):
        if sr == 0:
            self.sr = MAIN_SR
        else:
            self.sr = sr
        self.irs_path = IRS_PATH
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

    def pitch_shift(self, samples, semitones, granularity=5):
        shifted = rub.pyrb.pitch_shift(samples, self.sr, semitones, {'-c':granularity})
        return shifted

    def reverb(self, samples, rev_length):
        #convolution with randomly-selected impulse response
        #rev_length: int [1-4]
        analysis_path = os.path.join(IRS_PATH, 'ir_analysis.npy')
        ir_analysis = np.load(analysis_path, allow_pickle=True).item()
        if rev_length == 'any':
            rev_length = np.random.randint(5)
        chosen_irs = ir_analysis[rev_length]
        random_rev = random.choice(chosen_irs)

        rev_amount = np.random.rand() * 0.7 + 0.3

        IR, sr = librosa.core.load(random_rev, mono=False, sr=self.sr)

        try:  #convert to mono if sample is stereo
            IR = IR[0]
        except:
            pass

        IR = IR * rev_amount

        out = scipy.signal.convolve(samples, IR)
        #out = out / max(abs(out))
        #out = out * 0.9
        return out

    def reverb_stereo(self, samples_left, samples_right, rev_length='any'):
        #convolution with randomly-selected impulse response
        #rev_length: int [1-4]
        analysis_path = os.path.join(IRS_PATH, 'ir_analysis.npy')
        ir_analysis = np.load(analysis_path, allow_pickle=True).item()
        if rev_length == 'any':
            rev_length = np.random.randint(5)
        chosen_irs = ir_analysis[rev_length]
        random_rev = random.choice(chosen_irs)

        rev_amount = np.random.rand() * 0.7 + 0.3

        IR, sr = librosa.core.load(random_rev, mono=False, sr=self.sr)

        if len(IR.shape) == 2 and IR.shape[0] == 2:
            IR_final = IR
        else:
            IR_final = [IR,IR]

        IR_final = IR_final * rev_amount

        out_left = scipy.signal.convolve(samples_left, IR_final[0])
        out_right = scipy.signal.convolve(samples_right, IR_final[1])
        #out = out / max(abs(out))
        #out = out * 0.9
        return out_left, out_right

    def convolve(self, samples1, samples2):
        #convolve 2 signals
        out = scipy.signal.convolve(samples1, samples2)
        out = out / max(abs(out))
        out = out * 0.9
        return out

    def apply_fades(self, samples, in_fade_length, out_fade_length, exp=1):
        '''
        apply in and out fades to an input sound
        with exponential, log or linear curve (default linear)
        fade_lengths are in msecs
        '''
        in_fade_length = int(self.sr * in_fade_length / 1000)  #convert to samps
        out_fade_length = int(self.sr * out_fade_length / 1000)
        in_fade_length = np.clip(in_fade_length, 0, len(samples))
        out_fade_length = np.clip(out_fade_length, 0, len(samples))
        mask = np.ones(len(samples))
        ramp1 = np.arange(in_fade_length) / in_fade_length
        ramp2 = np.arange(out_fade_length) / (out_fade_length)
        ramp2 = np.array(np.flip(ramp2))
        ramp1 = ramp1 ** exp
        ramp2 = ramp2 ** exp
        mask[:in_fade_length] = ramp1
        mask[-out_fade_length:] = ramp2
        faded = samples * mask

        return faded

    def notch_filter(self, band, cutoff, ripple, rs, order=2, filter_type='cheby2'):
        #creates chebyshev polynomials for a notch filter with given parameters
        sr = self.sr
        nyq  = sr/2.0
        low  = cutoff - band/2.0
        high = cutoff + band/2.0
        low  = low/nyq
        high = high/nyq
        w0 = cutoff/(sr/2)
        a, b = iirfilter(order, [low, high], rp=ripple, rs=rs, btype='bandstop', analog=False, ftype=filter_type)

        return a, b

    def random_eq(self, vector_signal):
        #applies random filtering to an input vetor using chebyshev notch filters
        sr = self.sr
        num_filters = np.random.randint(1,4)
        #transition bands for cheby2 filter order
        low_edge = 80
        hi_edge = 8000
        cross1 = 200
        cross2 = 1000
        cross3 = 3000

        #deifine random parameters for 4 notch filters
        cutoff1 = np.random.randint(low_edge, hi_edge)
        band1 = np.random.randint(cutoff1/2+10, cutoff1-10)
        if cutoff1 >= low_edge and cutoff1<cross1:
            order1 = np.random.randint(2,3)
        elif cutoff1 >= cross1 and cutoff1<cross2:
            order1 = np.random.randint(2,4)
        elif cutoff1 >= cross2 and cutoff1<cross3:
            order1 = np.random.randint(2,5)
        elif cutoff1 >= cross1 and cutoff1<=hi_edge:
            order1 = np.random.randint(2,7)

        cutoff2 = np.random.randint(low_edge, hi_edge)
        band2 = np.random.randint(cutoff2/2+10, cutoff2-10)
        if cutoff2 >= low_edge and cutoff2<cross1:
            order2 = np.random.randint(2,3)
        elif cutoff2 >= cross1 and cutoff2<cross2:
            order2 = np.random.randint(2,4)
        elif cutoff2 >= cross2 and cutoff2<cross3:
            order2 = np.random.randint(2,5)
        elif cutoff2 >= cross1 and cutoff2<=hi_edge:
            order2 = np.random.randint(2,7)

        cutoff3 = np.random.randint(low_edge, hi_edge)
        band3 = np.random.randint(cutoff3/2+10, cutoff3-10)
        if cutoff3 >= low_edge and cutoff3<cross1:
            order3 = np.random.randint(2,3)
        elif cutoff3 >= cross1 and cutoff3<cross2:
            order3 = np.random.randint(2,4)
        elif cutoff3 >= cross2 and cutoff3<cross3:
            order3 = np.random.randint(2,5)
        elif cutoff3 >= cross1 and cutoff3<=hi_edge:
            order3 = np.random.randint(2,7)

        cutoff4 = np.random.randint(low_edge, hi_edge)
        band4 = np.random.randint(cutoff4/2+10, cutoff4-10)
        if cutoff4 >= low_edge and cutoff4<cross1:
            order4 = np.random.randint(2,3)
        elif cutoff4 >= cross1 and cutoff4<cross2:
            order4 = np.random.randint(2,4)
        elif cutoff4 >= cross2 and cutoff4<cross3:
            order4 = np.random.randint(2,5)
        elif cutoff4 >= cross1 and cutoff4<=hi_edge:
            order4 = np.random.randint(2,7)

        ripple = 10
        rs = 10

        #construct chebyshev notch filters
        a, b = self.notch_filter(band1,cutoff1,ripple, rs, order=order1)
        c, d = self.notch_filter(band2,cutoff2,ripple, rs, order=order2)
        e, f = self.notch_filter(band3,cutoff3,ripple, rs, order=order3)
        g, h = self.notch_filter(band4,cutoff4,ripple, rs, order=order4)

        #randomly concatenate 1,2,3 or 4 filters
        if num_filters == 1:
            filtered_data = lfilter(a, b, vector_signal)
        elif num_filters == 2:
            filtered_data = lfilter(a, b, vector_signal)
            filtered_data = lfilter(c, d, filtered_data)
        elif num_filters == 3:
            filtered_data = lfilter(a, b, vector_signal)
            filtered_data = lfilter(c, d, filtered_data)
            filtered_data = lfilter(e, f, filtered_data)
        elif num_filters == 4:
            filtered_data = lfilter(a, b, vector_signal)
            filtered_data = lfilter(c, d, filtered_data)
            filtered_data = lfilter(e, f, filtered_data)
            filtered_data = lfilter(g, h, filtered_data)

        return filtered_data


    def splitter(self, x, min_len=250):
        '''
        segment sound according to changes in the spectral shape
        return list of cut segments (diverse duration)
        '''

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

    def sel_good_segment(self, sound, dur, sr, perc=0.2):
        '''
        select a segment with duration similar to dur
        random chosing among the best candidates (perc%)
        '''
        clusters = self.splitter(sound)
        dists = []
        index = 0
        for i in clusters:
            cluster_len = len(i) / sr
            distance = np.abs(dur-cluster_len)
            dists.append((distance, index))
            index += 1

        sorted_by_dist = sorted(dists, key=lambda x: x[0])
        sorted_indexes = list(map(lambda x: x[1], sorted_by_dist))
        num_choices = int(np.ceil(len(sorted_indexes) * perc))
        choices = sorted_indexes[:num_choices]
        random_choice = random.choice(choices)
        random_sound = clusters[random_choice]

        return random_sound

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

    def distribute_pan_stereo(self, sounds, bounds=[0,1]):

        n_sounds = len(sounds)
        lens = []
        for i in sounds:
            lens.append(len(i))
        max_len = max(lens)
        #create ramp
        ramp = np.arange(n_sounds) / (n_sounds-1)
        #ramp = np.sqrt(ramp)  #better for panning
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

        out = np.divide(out, np.max(np.abs(out)))
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

    def load_split(self, soundfile, len_slices):
        sounds = []
        samples, sr = librosa.core.load(soundfile, self.sr)
        seg = np.arange(0, len(samples), len_slices)
        for i in seg:
            try:
                sounds.append(samples[i:i+len_slices])
            except:
                pass

        return sounds



    def concat_split(self, sounds, out_len, num_clusters, sil_perc_curve, sil_len_curve,
                            stretch_perc_curve, stretch_factor_curve, cluster_curve, fade_len=30,
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
            a = self.splitter(i, 250)
            for j in a:
                all_splits.append(j)
            uf.print_bar(index, len(sounds))
            index+=1
        len_out = 0

        output = self.gen_silence(0.1) #init out buf    fer with silence

        #re-order list clustering if wanted HERE!!
        if num_clusters == 1:
            clusters = {0: all_splits}
        else:
            clusters = self.cluster_data(all_splits, n_clusters=num_clusters)


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
                if num_clusters == 0:
                    curr_cluster = 0
                else:
                    curr_cluster = int(cluster_curve[curr_spot])

                if curr_cluster == 0:
                    #if curr_cluster is 0 select random cluster
                    sel_cluster = np.random.randint(num_clusters)
                else:
                    sel_cluster = curr_cluster-1
                random_i = np.random.randint(len(clusters[sel_cluster]))  #take random sound
                curr_sound = clusters[sel_cluster][random_i]

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

class Scene:
    '''
    dream scene.
    Global score contains all sounds
    '''
    def __init__(self, main_dur, sr=MAIN_SR, score_resolution=0.1):
        self.main_dur = main_dur  #scene duration in secs
        self.score_resolution = score_resolution  #in secs
        self.score_length = 1000
        self.global_score = {}  #all scores
        self.sr=sr
        self.post = Postprocessing(sr)

    def append_to_global_score(self, item, type, id):
        if id not in self.global_score.keys():  #create item if not exists
            self.global_score[id] = {}
        self.global_score[id][type] = item

    def plot_score(self, score_type='envelopes', dimensions=1):
        '''
        visualize score in 1,2 or 3 dimensions
        '''
        print ('plotting score')
        scores = []
        pans = []
        keys = self.global_score.keys()
        for i in keys:
            if score_type == 'envelopes':
                curr_sound = self.global_score[i]['samples']
                curr_score = np.clip(resample(np.abs(hilbert(curr_sound)), self.score_length), 0,1)
            elif score_type == 'squeres':
                curr_SCORE = self.global_score[i]['score']
            curr_pan = self.global_score[i]['pan'][0]
            scores.append(curr_score)
            pans.append(curr_pan)
        if dimensions == 1:
            for i in scores:
                plt.plot(i)
        if dimensions == 2:
            plt.pcolormesh(scores)
        if dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z=10
            ind = 0
            for ys in scores:
                xs = np.arange(len(ys))
                p = pans[ind]
                ax.bar(xs, ys, zs=p, zdir='y')
                z += 10
                ind += 1

            ax.set_xlabel('Time')
            ax.set_ylabel('Stereo')
            ax.set_zlabel('Volume')
            ax.set_ylim((-1,1))

        plt.show()

    def resolve_score_stereo(self, global_volume=1., fade_in=0, fade_out=0, global_rev=False, rev_length=4):
        '''
        takes avery file in the score and produces a stereo mix
        '''
        dur_samps = self.sr * self.main_dur
        mix_left = np.zeros(dur_samps)
        mix_right = np.zeros(dur_samps)


        #apply panning
        for sound in self.global_score.keys(): #iterate all sounds in score
            pan = self.global_score[sound]['pan'][0]

            samples = self.global_score[sound]['samples']
            angle = np.interp(pan, (-1,1), (-45,45))
            angle = np.radians(angle)
            left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * samples
            right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * samples
            mix_left = np.add(mix_left, left)
            mix_right = np.add(mix_right, right)


        #apply reverb
        if global_rev:
            mix_left, mix_right = self.post.reverb_stereo(mix_left, mix_right, rev_length)

        #normalize and rescale amplitude
        max_vol = max(max(abs(mix_left)),max(abs(mix_right)))
        mix_left = np.divide(mix_left, max_vol)
        mix_right = np.divide(mix_right, max_vol)

        if global_volume == 1.:
            mix_left = np.multiply(mix_left, 0.95)
            mix_right = np.multiply(mix_right, 0.95)
        else:
            mix_left = np.multiply(mix_left, global_volume)
            mix_right = np.multiply(mix_right, global_volume)


        #apply fades
        mix_left = self.post.apply_fades(mix_left, fade_in, fade_out, exp=1.3)  #apply fades
        mix_right = self.post.apply_fades(mix_right, fade_in, fade_out, exp=1.3)  #apply fades

        #concatenate channels
        mix = np.array([[mix_left,mix_right]])
        mix = np.squeeze(mix)

        return mix

    def resolve_score_quad(self):
        #TO BE TESTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        takes avery file in the score and produces a stereo mix
        '''
        dur_samps = self.sr * self.main_dur
        mix_fl = np.zeros(dur_samps)
        mix_fr = np.zeros(dur_samps)
        mix_bl = np.zeros(dur_samps)
        mix_br = np.zeros(dur_samps)

        for sound in self.global_score.keys(): #iterate all sounds in score
            pan_x = self.global_score[sound]['pan'][0]
            pan_y = self.global_score[sound]['pan'][1]
            samples = self.global_score[sound]['samples']
            angle_x = np.interp(pan_x, (-1,1), (-45,45))
            angle_y = np.interp(pan_y, (-1,1), (-45,45))
            angle_x = np.radians(angle_x)
            angle_y = np.radians(angle_y)

            left = np.sqrt(2)/2.0 * (np.cos(angle_x) - np.sin(angle_x)) * samples
            right = np.sqrt(2)/2.0 * (np.cos(angle_x) + np.sin(angle_x)) * samples

            fl = np.sqrt(2)/2.0 * (np.cos(angle_y) - np.sin(angle_y)) * left
            bl = np.sqrt(2)/2.0 * (np.cos(angle_y) + np.sin(angle_y)) * left
            fr = np.sqrt(2)/2.0 * (np.cos(angle_y) - np.sin(angle_y)) * right
            br = np.sqrt(2)/2.0 * (np.cos(angle_y) + np.sin(angle_y)) * right

            mix_fl = np.add(mix_fl, fl)
            mix_fr = np.add(mix_fr, fr)
            mix_bl = np.add(mix_bl, bl)
            mix_br = np.add(mix_br, br)
        mix = np.dstack((mix_fl,mix_fr,mix_bl,mix_br))[0]

        return mix

    def gen_onset_score(self, dur, volume, position):
        '''
        SINGLE SOUND ONSET SCORE
        array with zeros filled vith values (vol) like CV-GATE only in the onset time of a sound
        dur, volume in seconds, position is in percentage of the buffer
        '''
        num_steps = int(self.main_dur/self.score_resolution)  #total steps (length of entire scene)
        score = np.zeros(num_steps)  #empty score
        onset = np.ones(int(dur/self.score_resolution)) * volume  #onset segment
        offset = int(num_steps * position)
        if offset+len(onset) >= num_steps:  #cut onset if exceeding length
            cut_len = (num_steps - offset)
            onset = onset[:cut_len]
        score[offset:offset+len(onset)] = onset  #fill score with the onset

        return score

    def get_sound(self, category, model, variation, dur):
        '''
        get a random sound within in the specified folder
        '''
        sounds_path = os.path.join(SRNN_DATA_PATH, category, model,
                                   'sounds', 'dur_' + str(dur), 'model_' + str(variation))
        sounds_path = os.path.abspath(sounds_path)
        sounds = os.listdir(sounds_path)
        sounds = list(filter(lambda x: '.wav' in x, sounds))
        chosen = random.choice(sounds)
        chosen_path = os.path.abspath(os.path.join(sounds_path, chosen))
        samples, sr = librosa.core.load(chosen_path, sr=self.sr)

        return samples

    def gen_sound_from_parameters(self, parameters, overwrite=False, verbose=False):
        '''
        select sound from parameters dict
        '''
        #if an overwrite dict is supplied, overwrite only the contained keys
        if isinstance(overwrite, dict):
            p = sc.get_constrains([parameters, overwrite])
        else:
            p = parameters

        if verbose:
            print ('category: ' + str(p['sound']['category']))
            print ('model: ' + str(p['sound']['model']))
            print ('variation: ' + str(p['sound']['variation']))
            print ('dur: ' + str(p['sound']['dur']))


        sound = self.get_sound(category=p['sound']['category'],
                               model=p['sound']['model'],
                               variation=p['sound']['variation'],
                               dur=p['sound']['dur'])

        return sound

    def score_sound_from_parameters(self, sound, parameters, id, overwrite=False, verbose=False):
        '''
        build sound from parameters dict
        '''
        #if an overwrite dict is supplied, overwrite only the contained keys
        if isinstance(overwrite, dict):
            p = sc.get_constrains([parameters, overwrite])
        else:
            p = parameters

        self.gen_scored_sound(sound=sound,
                         dur=p['score']['dur'],
                         volume=p['score']['volume'],
                         position=p['score']['position'],
                         pan=p['score']['pan'],
                         eq=p['score']['eq'],
                         rev=p['score']['rev'],
                         rev_length=p['score']['rev_length'],
                         segment=p['score']['segment'],
                         stretch=p['score']['stretch'],
                         shift=p['score']['shift'],
                         fade_in=p['score']['fade_in'],
                         fade_out=p['score']['fade_out'],
                         id=id,
                         verbose=verbose)


    def gen_scored_sound(self, sound,  dur, volume, position, pan,
                        eq=False, rev=False, rev_length=4, segment=False,
                        stretch=1, shift=0, fade_in=20, fade_out=100, id=0,
                        verbose=False):
        '''
        compute sound and apply all processings. Append to global score
        dur: float, seconds
        volume: float [0-1]
        position: (time_position) float[0-1], percentage of all time
        pan = tuple, floats[-1-1], first is horizontal, second is diagonal
        eq: bool, apply random eq
        rev: bool, aplly random rev
        rev_type: string, reverb type
        segment: bool, extract 1 clustered segment
        stretch: float, stretch factor, apply time_stretching
        shift:float, semitones, apply pitch_shift
        fade in/out: int, milliseconds
        id: int, sound id for the global score
        '''
        if verbose:
            print('loading')
        pad = np.zeros(self.sr * self.main_dur) #inital pad, long as the whole score
        num_samps = len(pad)
        offset = int(num_samps * position)
        dur_samps = int(dur * self.sr)
        sound = sound / np.max(sound)  * volume #normalize and rescale sound

        if verbose:
            print('segmentation')
        #apply eventual processing
        if segment:  #clustering-based segmentation
            #select the segment with the most similar duration to the
            #score
            sound = self.post.sel_good_segment(sound, dur, self.sr, perc=0.2)
            sound = self.post.apply_fades(sound, 20, 50, exp=1.3)

        if verbose:
            print('stretching')
        if stretch != 1:
            sound = self.post.stretch(sound, stretch)

        if verbose:
            print('eqing')
        if eq:
            sound = self.post.random_eq(sound)

        if verbose:
            print('reverberation')
        if rev:
            sound = self.post.reverb(sound, rev_length)

        if verbose:
            print('cutting')
        if len(sound) < dur_samps:  #shorten score length is sample is too short
            dur_samps = int(len(sound))
        else:
            sound = sound[:dur_samps]  #cut sound if too long

        if offset+dur_samps >= num_samps:  #cut offset if exceeding length
            dur_samps = (num_samps - offset)
            sound = sound[:dur_samps]

        if verbose:
            print('shifting')
        if shift != 0:
            sound = self.post.pitch_shift(sound, shift)

        if verbose:
            print('fading')
        sound = self.post.apply_fades(sound, fade_in, fade_out, exp=1.3)  #apply fades

        pad[offset:offset+len(sound)] = sound

        sound = pad

        score = self.gen_onset_score(len(sound), volume, position)

        self.append_to_global_score(score, 'score', id)
        self.append_to_global_score(pan, 'pan', id)
        self.append_to_global_score(sound, 'samples', id)

        return sound

    def gen_macro(self, verbose=False):
        constrains_dict = sc.gen_random_macro(verbose=verbose)

        return constrains_dict

    def gen_random_parameters(self, constrains='None', fixed_category=False,
                              fixed_model=False, verbose=False):
        '''
        generate random parameters to build a scene
        everything is based on uniform random choice within a list of parameters
        'constrains' modify the lists to bias decisions
        '''
        #macro categories of parameters
        if constrains == 'None':
            constrains = {'sound':{},
                          'score':{}
                          }

        parameters = {'sound':{},
                      'score':{}
                      }

        parameters['sound'] = {'category': [],
                               'model': [],
                               'variation': [],
                               'dur': []
                               }

        parameters['score'] = {'dur': [],
                               'volume': [],
                               'position': [],
                               'pan': [],
                               'eq': [],
                               'rev': [],
                               'rev_length': [],
                               'segment': [],
                               'stretch': [],
                               'shift': [],
                               'fade_in': [],
                               'fade_out': [],
                               }


        #SOUND PARAMETERS
        #category
        availables = sc.check_available_models()

        if fixed_category:
            sel_category = fixed_category
        else:
            categories = list(availables.keys())
            if 'category' in constrains['sound'].keys():
                categories = constrains['sound']['category'](categories)
                #print ('constrain for sound, category: ' + str(constrains['sound']['category']))
            else:
                pass
                #print ('no constrain for sound, category')
            sel_category = np.random.choice(categories)
        parameters['sound']['category'] = sel_category
        if verbose:
            print ('chosen parameter for sound, category: ' + str(sel_category))

        #model
        if fixed_model:
            sel_model = fixed_model
        else:
            models = list(availables[sel_category])
            if 'model' in constrains['sound'].keys():
                models = constrains['sound']['model'](models)
                #print ('constrain for sound, model: ' + str(constrains['sound']['model']))
            else:
                pass
                #print ('no constrain for sound, model')
            sel_model = np.random.choice(models)
        parameters['sound']['model'] = sel_model
        if verbose:
            print ('chosen parameter for sound, model: ' + str(sel_model))

        #variation
        variations = list(models_map[sel_category][sel_model].keys())
        if 'variation' in constrains['sound'].keys():
            variations = constrains['sound']['variation'](variations)
            #print ('constrain for sound, variation: ' + str(constrains['sound']['variation']))
        else:
            pass
            #print ('no constrain for sound, variation')
        sel_variation = np.random.choice(variations)
        parameters['sound']['variation'] = sel_variation
        if verbose:
            print ('chosen parameter for sound, variation: ' + str(sel_variation))

        #dur
        durations = list(durations_map.keys())
        if 'dur' in constrains['sound'].keys():
            durations = constrains['sound']['dur'](durations)
            #print ('constrain for sound, dur: ' + str(constrains['sound']['dur']))
        else:
            pass
            #print ('no constrain for sound, dur')
        sel_dur = np.random.choice(durations)
        parameters['sound']['dur'] = sel_dur
        if verbose:
            print ('chosen parameter for sound, dur: ' + str(sel_dur))

        #SCORE PARAMETERS
        #score_duration
        score_durations = np.arange(0, self.main_dur, 0.01)  #10 ms resolution
        if 'dur' in constrains['score'].keys():
            score_durations = constrains['score']['dur'](score_durations)
            #print ('constrain for score, dur: ' + str(constrains['score']['dur']))
        else:
            pass
            #print ('no constrain for score, dur')
        sel_score_dur = np.random.choice(score_durations)
        parameters['score']['dur'] = sel_score_dur
        if verbose:
            print ('chosen parameter for score, dur: ' + str(sel_score_dur))


        #volume
        volumes = np.arange(0., 1., 0.01)  #10 ms resolution
        if 'volume' in constrains['score'].keys():
            volumes = constrains['score']['volume'](volumes)
            #print ('constrain for score, volume: ' + str(constrains['score']['volume']))
        else:
            pass
            #print ('no constrain for score, volume')
        sel_volume = np.random.choice(volumes)
        parameters['score']['volume'] = sel_volume
        if verbose:
            print ('chosen parameter for score, volume: ' + str(sel_volume))

        #position
        positions = np.arange(0, 1, 0.01)  #10 ms resolution
        if 'position' in constrains['score'].keys():
            positions = constrains['score']['position'](positions)
            #print ('constrain for score, position: ' + str(constrains['score']['position']))
        else:
            pass
            #print ('no constrain for score, position')
        sel_position = np.random.choice(positions)
        parameters['score']['position'] = sel_position
        if verbose:
            print ('chosen parameter for score, position: ' + str(sel_position))

        #pan  only stereo for now
        pans = np.arange(-1, 1, 0.01)  #10 ms resolution
        if 'pan' in constrains['score'].keys():
            pans = constrains['score']['pan'](pans)
            #print ('constrain for score, pan: ' + str(constrains['score']['pan']))
        else:
            pass
            #print ('no constrain for score, pan')
        sel_pan = np.random.choice(pans)
        parameters['score']['pan'] = [sel_pan, sel_pan]
        if verbose:
            print ('chosen parameter for score, pan: ' + str(sel_pan))

        #eq
        eqs = [True, False]  #10 ms resolution
        if 'eq' in constrains['score'].keys():
            eqs = constrains['score']['eq'](eqs)
            #print ('constrain for score, eq: ' + str(constrains['score']['eq']))
        else:
            pass
            #print ('no constrain for score, eq')
        sel_eq = np.random.choice(eqs)
        parameters['score']['eq'] = sel_eq
        if verbose:
            print ('chosen parameter for score, eq: ' + str(sel_eq))

        #rev

        revs = [True, False]  #10 ms resolution
        ass = lambda x: [False, True, False]

        if 'rev' in constrains['score'].keys():
            revs = constrains['score']['rev'](revs)
            #print ('constrain for score, rev: ' + str(constrains['score']['rev']))
        else:
            pass
            #print ('no constrain for score, rev')
        sel_rev = np.random.choice(revs)
        parameters['score']['rev'] = sel_rev
        if verbose:
            print ('chosen parameter for score, rev: ' + str(sel_rev))

        #rev_length
        analysis_path = os.path.join(IRS_PATH, 'ir_analysis.npy')
        rev_analysis_file = np.load(analysis_path, allow_pickle=True).item()
        num_lengths = len(list(rev_analysis_file.keys()))
        rev_lengths = np.arange(num_lengths)  #10 ms resolution
        if 'rev_length' in constrains['score'].keys():
            rev_lengths = constrains['score']['rev_length'](rev_lengths)
            #print ('constrain for score, rev_length: ' + str(constrains['score']['rev_length']))
        else:
            pass
            #print ('no constrain for score, rev_length')
        sel_rev_length = np.random.choice(rev_lengths)
        parameters['score']['rev_length'] = sel_rev_length
        if verbose:
            print ('chosen parameter for score, rev_length: ' + str(sel_rev_length))

        #segment
        segments = [True, False]  #10 ms resolution
        if 'segment' in constrains['score'].keys():
            segments = constrains['score']['segment'](segments)
            #print ('constrain for score, segment: ' + str(constrains['score']['segment']))
        else:
            pass
            #print ('no constrain for score, segment')
        sel_segment = np.random.choice(segments)
        parameters['score']['segment'] = sel_segment
        if verbose:
            print ('chosen parameter for score, segment: ' + str(sel_segment))

        #stretch
        stretches1 = np.arange(0.1, 1, 0.01) ** 2
        stretches2 = np.arange(1, 3, 0.02) ** 2
        stretches = np.concatenate((stretches1, stretches2))
        if 'stretch' in constrains['score'].keys():
            stretches = constrains['score']['stretch'](stretches)
            #print ('constrain for score, stretch: ' + str(constrains['score']['stretch']))
        else:
            pass
            #print ('no constrain for score, stretch')
        sel_stretch = np.random.choice(stretches)
        parameters['score']['stretch'] = sel_stretch
        if verbose:
            print ('chosen parameter for score, stretch: ' + str(sel_stretch))

        #shift
        shifts = np.arange(-48, 24, 0.1)
        if 'shift' in constrains['score'].keys():
            shifts = constrains['score']['shift'](shifts)
            #print ('constrain for score, shift: ' + str(constrains['score']['shift']))
        else:
            pass
            #print ('no constrain for score, shift')
        sel_shift = np.random.choice(shifts)
        parameters['score']['shift'] = sel_shift
        if verbose:
            print ('chosen parameter for score, shift: ' + str(sel_shift))

        #fade_in
        fade_ins = np.arange(0, sel_score_dur*1000/2,1)
        if 'fade_in' in constrains['score'].keys():
            fade_ins = constrains['score']['fade_in'](fade_ins)
            #print ('constrain for score, fade_in: ' + str(constrains['score']['fade_in']))
        else:
            pass
            #print ('no constrain for score, fade_in')
        sel_fade_in = np.random.choice(fade_ins)
        parameters['score']['fade_in'] = sel_fade_in
        if verbose:
            print ('chosen parameter for score, fade_in: ' + str(sel_fade_in))

        #fade_out
        fade_outs = np.arange(0, sel_score_dur*1000/2,1)
        if 'fade_out' in constrains['score'].keys():
            fade_out = constrains['score']['fade_out'](fade_outs)
            #print ('constrain for score, fade_out: ' + str(constrains['score']['fade_out']))
        else:
            pass
            #print ('no constrain for score, fade_out')
        sel_fade_out = np.random.choice(fade_outs)
        parameters['score']['fade_out'] = sel_fade_out
        if verbose:
            print ('chosen parameter for score, fade_out: ' + str(sel_fade_out))


        return parameters



class BuildScene:
    '''
    build dream scene.
    Global score contains all sounds
    '''
    def __init__(self, max_dur=60, max_num_sounds = 100, sr=MAIN_SR):
        self.max_num_sounds = max_num_sounds
        self.max_dur = 60
        self.score_resolution = 0.1  #in secs
        self.score_length = 1000
        self.sr=sr

    def build(self, length, density, score_diversity, sel_diversity, single_model=False,
              fixed_category='rand', fixed_model='rand', fast=True, carpet=True, verbose=False):
        '''
        generate scene from macroparameters
        fast= no shift, no stretch
        carpet = 1 or 2 long sounds starting from the berginning
        '''
        #scale variables by macroparameters
        random_diversity_flag = random.choice([True, False])  #50% choice
        carpet_num = random.choice([1,2])
        print (random_diversity_flag)
        scene_dur = int(np.ceil(self.max_dur * length))
        num_sounds = int(np.ceil(self.max_num_sounds * density))
        different_sounds = int(np.ceil(num_sounds * sel_diversity))
        different_scores = int(np.ceil(num_sounds * score_diversity))
        scene = Scene(main_dur=scene_dur, sr=self.sr)
        sound_macros = {}
        score_macros = {}

        #compute sound and scene macros
        if verbose:
            print ('sound macros:')

        for i in range(different_sounds):
            curr_macro = scene.gen_macro(verbose=False)
            sound_macros[i] = curr_macro
        if verbose:
            print ('scene macros:')

        for i in range(different_scores):
            curr_macro = scene.gen_macro(verbose=False)
            score_macros[i] = curr_macro

        #dealing with sound selection
        if single_model:
            ava = check_available_models()
            if fixed_category == 'rand':
                cats = list(ava.keys())
                ch_category = random.choice(cats)
            else:
                ch_category = fixed_category

            if fixed_model == 'rand':
                ch_model = random.choice(ava[ch_category])
            else:
                ch_model = fixed_model

        if not single_model and random_diversity_flag:
            print ('vaffanculo')
            ava = sc.check_available_models()
            possible_categories = []
            possible_models = []
            for i in range(different_sounds):
                cats = list(ava.keys())
                ch_category = random.choice(cats)
                ch_model = random.choice(ava[ch_category])
                possible_categories.append(ch_category)
                possible_models.append(ch_model)

        #building dictionary of fixed options
        options = copy.deepcopy(sc.constrains_dict)

        if fast:
            options['score']['shift'] = 0
            options['score']['stretch'] = 1


        #build_scene
        print ('building scene')
        index = 0
        for i in range(num_sounds):
            #choose random macro (dict of constrains) from the available ones
            rand_sound_macro = random.choice(list(sound_macros.keys()))
            rand_score_macro = random.choice(list(score_macros.keys()))
            curr_sound_macro = sound_macros[rand_sound_macro]
            curr_score_macro = score_macros[rand_score_macro]

            #generate random prameters with chosen constrains
            #if not single sound 50% times section diversity choses also the number of sounds
            if single_model:
                #sound is fixed
                curr_sound_parameters = scene.gen_random_parameters(fixed_category=ch_category,
                                                                    fixed_model=ch_model)
            else:
                if random_diversity_flag:
                    #number of possible models is connected to sel_diversity
                    random_sel = np.random.randint(len(possible_categories))
                    ch_category = possible_categories[random_sel]
                    ch_model = possible_models[random_sel]
                    curr_sound_parameters = scene.gen_random_parameters(fixed_category=ch_category,
                                                                        fixed_model=ch_model)
                else:
                    #model selection is completely random
                    curr_sound_parameters = scene.gen_random_parameters()

            curr_score_parameters = scene.gen_random_parameters(curr_score_macro)


            if carpet:
                if i <= carpet_num:
                    options = se.get_constrains(options, 'long_')  vaffanculo
                    options['score']['length'] = 1



            #compute sound
            curr_sound = scene.gen_sound_from_parameters(curr_sound_parameters, overwrite=options, verbose=False)

            #post processing and put sound into score
            scene.score_sound_from_parameters(curr_sound, curr_score_parameters, i,
                                               overwrite=options, verbose=False)

            uf.print_bar(index, num_sounds)
            index += 1










        print ('fottiti')
