import numpy as np
import utility_functions as uf
from scipy import signal
from scipy.signal import iirfilter, lfilter, convolve
#from librosa.core import load as audio_load
#import librosa.effects as eff
import librosa
from shutil import copyfile
import os, random, sys
import loadconfig
import essentia.standard as ess
import configparser
import matplotlib.pyplot as plt

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
SR =  cfg.getint('sampling', 'sr_target')
AUGMENTATION_IN = sys.argv[1]
AUGMENTATION_OUT = sys.argv[2]
NUM_EXTENSIONS = int(sys.argv[3])
'''
AUGMENTATION_IN = cfg.get('augmentation', 'augmentation_in')
AUGMENTATION_OUT = cfg.get('augmentation', 'augmentation_out')
NUM_EXTENSIONS = cfg.getint('augmentation', 'num_extensions')
'''
IR_FOLDER = cfg.get('augmentation', 'augmentation_IRs_path')
NOISE_SAMPLE = cfg.get('augmentation', 'augmentation_backgroundnoise_path')
#get samples length
temp_contents = os.listdir(AUGMENTATION_IN)
temp_contents = list(filter(lambda x: '.wav' in x, temp_contents))
in_file_name = AUGMENTATION_IN + '/' + temp_contents[0]
global_sr, dummy = uf.wavread(in_file_name)


#load background noise sample
print ("loading background noise sample...")

#background_noise1, nsr = audio_load(NOISE_SAMPLE, sr=SR)
noise_loader = ess.EasyLoader(filename=NOISE_SAMPLE, sampleRate=global_sr)
background_noise = noise_loader()

def notch_filter(band, cutoff, ripple, rs, sr=global_sr, order=2, filter_type='cheby2'):
    #creates chebyshev polynomials for a notch filter with given parameters
    nyq  = sr/2.0
    low  = cutoff - band/2.0
    high = cutoff + band/2.0
    low  = low/nyq
    high = high/nyq
    w0 = cutoff/(sr/2)
    a, b = iirfilter(order, [low, high], rp=ripple, rs=rs, btype='bandstop', analog=False, ftype=filter_type)

    return a, b

def bg_noise(vector_signal, dur):
    tmp_noise = np.array([])
    #concatenate random noise chunks until vector is bigger than input
    while len(tmp_noise) <= dur:
        inchunk = np.random.randint(dur/2)
        outchunk = np.random.randint(dur/2+dur)
        tmp_noise = np.concatenate((tmp_noise, background_noise[inchunk:outchunk]))
    max_bound = len(tmp_noise) - 100
    rand_start = np.random.randint(max_bound)  #random position in noise sample
    noise_chunk = np.array(tmp_noise[:dur])  #extract noise chunk

    wet_gain = ((np.random.random_sample()*0.5)+0.5) * 0.25
    dry_gain = 1.0 - wet_gain
    mix_output = np.add(vector_signal*dry_gain, noise_chunk*wet_gain)

    return mix_output


def random_eq(vector_signal, dur, sr=global_sr):
    #applies random filtering to an input vetor using chebyshev notch filters
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
    a, b = notch_filter(band1,cutoff1,ripple, rs, order=order1)
    c, d = notch_filter(band2,cutoff2,ripple, rs, order=order2)
    e, f = notch_filter(band3,cutoff3,ripple, rs, order=order3)
    g, h = notch_filter(band4,cutoff4,ripple, rs, order=order4)

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

def random_stretch(vector_signal, dur, sr=SR):
    #applies random time stretch to signal

    #dur_samps = int(sr*dur)
    stretched_vector = []

    random_rate = ((np.random.randint(1, 50)/50.)-0.5)*0.04 #random time stretch rate
    rate = 1 + random_rate
    output_vector = np.zeros(dur*2)
    #vector_signal = np.concatenate((vector_signal, vector_signal))  #double the signal vector
    #vector_signal = eff.time_stretch(vector_signal, rate)  #compute time stretch (bad algo)

    indexes = np.round(np.arange(0, len(vector_signal)-1, rate)).astype(int)
    for i in indexes:
    	stretched_vector.append(vector_signal[i])

    output_vector[:len(stretched_vector)] = stretched_vector
    output_vector = output_vector[:dur]  #cut sound over dur

    return output_vector



def random_rev(vector_signal, dur):
    #process a sound convolving it with a random reverb Impulse Response
    IR_random = 0
    while IR_random == 0:  #iterate until a wav file is chosen
        IR_choice = random.choice(os.listdir(IR_FOLDER))  #select random IR from folder
        if IR_choice[-3:] == "wav":  #check if is a wav file
            IR_random = IR_choice

    IR_filename = IR_FOLDER + '/' + IR_random  #reconstruct IR path
    IR_loader = ess.EasyLoader(filename=IR_filename, sampleRate=SR, downmix='left')
    IR = IR_loader()
    #sr, IR = uf.wavread(IR_filename)  #read IR file
    #IR, srate = audio_load(IR_filename, sr=global_sr)  #read and resample
    #IR = IR[:,0]  #take just left channel of the stereo file
    #dur_samps = int(sr*dur)  #duration in samples
    convoluted = convolve(vector_signal,IR)  #convolution
    convoluted = convoluted/np.max(np.abs(convoluted))  #normalization
    convoluted_signal = convoluted[:dur]  #cut the tail of the convoluted sound
    #mix dry and wet signals
    wet_gain = ((np.random.random_sample()*0.5)+0.5) * 0.08
    dry_gain = 1.0 - wet_gain
    mix_output = np.add(vector_signal*dry_gain, convoluted_signal*wet_gain)

    return mix_output



def random_samples(vector_signal, dur, sr=SR):
    '''
    num_rand_samps = np.random.randint(1,3)
    #dur_samps = int(sr*dur)
    for sample in range(num_rand_samps):
        rand_pos = np.random.randint(0, dur-1)
        rand_samp = np.random.sample()
        vector_signal[rand_pos] = rand_samp
    '''

    return vector_signal


def extend_datapoint(file_name, output_dir=AUGMENTATION_OUT, num_extensions=1, status = [1,0]):
    #creates alternative versions of sounds of a single sound trying to keep the chaos/order feature

    sound_name = file_name.split('/')[-1]
    sound_string = sound_name[:-4]
    label_string = sound_name[-4:]
    label = label_string[1]
    sr, vector_input = uf.wavread(file_name)
    DUR = len(vector_input)
    funcs = ['random_stretch','bg_noise','random_eq']

    for new_sound in range(num_extensions):

        np.random.shuffle(funcs) #scramble order of functions
        rev_prob = np.random.randint(1,3)  #1/3 files will have reverb
        rand_samp_prob = np.random.randint(1,3)  #1/3 files will have random samples
        num_nodes = np.random.randint(1,3)  #nodes probability
        random_appendix = np.random.randint(10000)  #random number to append to filename (so.. validation split of dataset will be composed of random sounds)

        node1 = 'node1_out = ' + funcs[0] + '(vector_input, dur=DUR)'
        node2 = 'node2_out = ' + funcs[1] + '(node1_out, dur=DUR)'
        node3 = 'node3_out = ' + funcs[2] + '(node2_out, dur=DUR)'

        if num_nodes == 1:
            exec(node1)
            vector_output = locals()['node1_out']
        if num_nodes == 2:
            exec(node1)
            exec(node2)
            vector_output = locals()['node2_out']
        if num_nodes == 3:
            exec(node1)
            exec(node2)
            exec(node3)
            vector_output = locals()['node3_out']

        if rev_prob == 1:
            vector_output = random_rev(vector_output, dur=DUR)
        '''
        if rand_samp_prob == 1:
            vector_output = random_samples(vector_output, dur=DUR)
        '''

        #output_normalization
        vector_output = np.divide(vector_output, np.max(vector_output))
        vector_output = np.multiply(vector_output, 0.8)
        #formatting strings to print
        success_string = sound_string + ' augmented: ' + str(new_sound+1)  #describe last prcessed sound
        infolder_num_files = status[0]  #number of input files to extend
        current_batch = status[1]  #count of processed input files
        total_num_files = infolder_num_files * num_extensions  #total number of datapoint extension files to create
        current_processed_file = (num_extensions * current_batch) + (new_sound + 1)  #number of currently processed files
        perc_progress = (current_processed_file * 100) / total_num_files  #compute percentage of progress
        status_string = 'status: ' + str(perc_progress) + '% | ' + 'processed files: ' + str(current_processed_file) + '/' + str(total_num_files)  #format progress string

        sound_name = sound_name.split('.')[0]
        output_file_name = output_dir + '/' + sound_name + '.' + '.augmented_' + str(new_sound+1) + '.mp4.wav'  #build output file name
        uf.wavwrite(vector_output, global_sr, output_file_name)   #create output file
        #print progress and status strings
        print(success_string)
        print(status_string)


def main(num_extensions, input_dir, output_dir):
    #creates alternative versions of sounds of an entire dataset trying to keep the chaos/order feature
    contents = os.listdir(input_dir)

    num_sounds = len(list(filter(lambda x: x[-3:] == "wav", contents)))

    count = 0  #processed input files count
    for file in os.listdir(input_dir):
        if file[-3:] == "wav": #takes just .wav files
            in_file = os.path.join(input_dir, file )
            output_filename = file.split('.')[0] + '.original.wav'
            out_file = os.path.join(output_dir, output_filename)
            sr, original_sound = uf.wavread(in_file)
            #normalize original sound
            original_sound = np.divide(original_sound, np.max(original_sound))
            original_sound = np.multiply(original_sound, 0.9)
            #copyfile(in_file, out_file)
            uf.wavwrite(original_sound, global_sr, out_file)
            status = [num_sounds, count]  #pass to extend_datapoint() input folder numsounds and current precessei infiles count
            extend_datapoint(file_name=in_file, output_dir=output_dir, num_extensions=num_extensions, status=status)  #create extension files

            count = count + 1  #progress input files count

    print('Dataset successfully augmented from ' + str(num_sounds) + ' to ' + str(num_sounds * num_extensions + num_sounds) + ' sounds')

if __name__ == '__main__':
    main(NUM_EXTENSIONS, AUGMENTATION_IN, AUGMENTATION_OUT)
