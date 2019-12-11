import numpy as np
import utility_functions as uf
import postprocessing_utils as pp
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import librosa
import scipy
from modules import *
import sys

post = Postprocessing(16000, '../IRs/revs/divided/')

s1 = '/home/eric/Downloads/voice_prova.wav'


out_file = '/home/eric/Desktop/quick_sounds/CULISMO'
dur = 16000
sounds = []
samples, sr = librosa.core.load(s1)
seg = np.arange(0, len(samples), dur)

for i in seg:
        curr = samples[i:i+dur]
        if len(curr) == dur:
            #sounds.append(np.abs(scipy.fftpack.fft(samp, 128)[1:64]))
            sounds.append(curr)
        #sounds.append([np.random.randint(10),np.random.randint(10)])
#sounds = np.array(sounds)

def cluster_data(sounds, n_clusters=5):
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

    for i in range(n_clusters):
        print (i, len(output[i]))





cluster_data(sounds, 5)




#clustering = DBSCAN(eps=9, min_samples=2).fit(sounds)
