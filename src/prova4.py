import numpy as np
import matplotlib.pyplot as plt
import librosa
from modules import Postprocessing
post = Postprocessing(sr=44100)
sr=44100

a = '/Users/eric/Desktop/memories/shared/coglionazzo/0.wav'

input_vector, sr  = librosa.core.load(a, sr=44100)
'''
input_vector[3*sr:10*sr] = 0
input_vector[14*sr:20*sr] = 0
input_vector[25*sr:27*sr] = 0
input_vector[-10*sr:] = 0
'''





cut_silence_multichannel(input_vector)
