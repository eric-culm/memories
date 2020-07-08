import librosa
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqs

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs=44100, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def b_filter(input_vector, cutoff=2000, fs=44100, order=16):
    '''
    backward lowpass filter
    '''
    #input_vector = input_vector[-1:0]
    stft = librosa.core.stft(input_vector)
    mags = np.abs(stft)

    filtered = butter_lowpass_filter(input_vector, cutoff, order=order)
    stft_filt = librosa.core.stft(filtered)
    mags_filt = np.abs(stft_filt)

    flipped = np.flip(input_vector)
    flipped_filt = butter_lowpass_filter(flipped, cutoff, order=order)
    reflipped = np.flip(flipped_filt)

    stft_flip = librosa.core.stft(reflipped)
    mags_flip = np.abs(stft_flip)


    plt.subplot(311)
    plt.pcolormesh(mags**0.5/3)
    plt.title('normal')
    plt.subplot(312)
    plt.pcolormesh(mags_filt**0.5/3)
    plt.title('filtered')
    plt.subplot(313)
    plt.pcolormesh(mags_flip**0.5/3)
    plt.title('flipped')
    plt.show()



if __name__ == '__main__':
    file_path = '/Users/eric/Desktop/quick_sounds/lamine_5.wav'
    samples, sr = librosa.core.load(file_path, sr=44100, mono=False)
    b_filter(samples)
