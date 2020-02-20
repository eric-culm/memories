import numpy as np
import matplotlib.pyplot as plt
import librosa
from modules import *
import threading
post = Postprocessing(sr=44100)
#scene = Scene(main_dur=60)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()
build = BuildScene()
dream = Dream()

dream.random_dream(2*60)
