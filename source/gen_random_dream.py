import numpy as np
import matplotlib.pyplot as plt
import librosa
from modules import *
import threading
post = Postprocessing(sr=44100)
#scene = Scene(main_dur=60)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
#post = Postprocessing()
build = BuildScene()
#dream = Dream(scene_maxdur=60, max_num_sounds=50)
choice_dict = {'instrumental': ['pianoDreamy']}

dream = Dream(scene_maxdur=60, max_num_sounds=50)

mix = dream.random_dream(1*60, neuro_choice=choice_dict)
print("cwefdf")
#allocator.write_local(mix, 'booooooo')
