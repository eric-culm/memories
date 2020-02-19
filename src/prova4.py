import numpy as np
import matplotlib.pyplot as plt
import librosa
from modules import Postprocessing
post = Postprocessing(sr=44100)
sr=44100

a = '/Users/eric/Desktop/memories/shared/coglionazzo/0.wav'

input_vector, sr  = librosa.core.load(a, sr=44100)
input_vector[3*sr:10*sr] = 0
input_vector[14*sr:20*sr] = 0



def cut_silence_multichannel(input_vector, max_sil_len=3):
    '''
    cut silence from the beginning, end of a multichannel audio file
    and cut silence in the middle if it is longer than max_sil_len
    '''

    if len(input_vector.shape) == 1: #if mono file
        input_vector = np.array([input_vector, input_vector])  #copy to Stereo

    mono_vec = np.sum(input_vector, axis=0) / np.max(input_vector)

    split_vec = librosa.effects.split(mono_vec, top_db = 60)
    onset = split_vec[0][0]
    offset = split_vec[-1][-1]
    offset = 5000
    print (input_vector.shape)
    #cut init and final silences



    for channel in range(len(input_vector)):
        input_vector[channel] = input_vector[channel][onset:offset] # cut beginning and ending silence

    plt.subplot(221)
    plt.plot(input_vector[0])
    plt.subplot(222)
    plt.plot(input_vector[1])
    plt.show()


    #cut intermediate silences longer than max_sil_len
    #list of silence positions to be cut
    cuts_list = []
    if len(split_vec) > 1:
        for i in range(len(split_vec)-1):
            curr_end = split_vec[i][1]
            next_start = split_vec[i+1][0]
            dist = (next_start - curr_end) / sr
            if dist > max_sil_len:
                cuts_list.append([curr_end, next_start])

        #add new reduced silence
        for k in range(len(cuts_list)):
            len_new_silence = int(np.random.uniform() * max_sil_len * sr) #random silence time
            len_new_silence = int(np.clip(len_new_silence, sr/2, max_sil_len * sr))
            cuts_list[k][0] = cuts_list[k][0] + len_new_silence

        #build output vector
        output_vector = np.empty(input_vector.shape)
        for channel in range(len(input_vector)):
            output_vector[channel] = input_vector[channel][:cuts_list[0][0]]
            for cut in range(len(cuts_list)-1):
                output_vector[channel] = post.xfade(output_vector[channel], input_vector[channel][cuts_list[cut][1]:cuts_list[cut+1][0]], 2000)
            output_vector[channel] = post.xfade(output_vector[channel], input_vector[channel][cuts_list[-1][1]:], 2000)
    plt.subplot(223)
    plt.plot(output_vector[0])
    plt.subplot(224)
    plt.plot(output_vector[1])
    #plt.subplot(212)
    #plt.plot(output_vector)
    plt.show()

cut_silence_multichannel(input_vector)
