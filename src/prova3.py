import matplotlib.pyplot as plt
import numpy as np
from modules import *
import scene_constrains as sc
import random
import utility_functions as uf

scene = Scene(main_dur=30)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

#constrains = only_available

num_sounds = 1000
index = 0
for i in range(num_sounds):
    if index in [0,1,2,3,4]:
        constrains = sc.get_constrains(['only_available', 'long_low', 'no_stretch', 'at_beginning', 'volume_hi'])
    else:
        constrains = sc.get_constrains(['only_available', 'long_low', 'no_stretch'])
    constrains = sc.get_constrains(['only_available', 'long_low', 'no_stretch'])
    p = scene.gen_random_parameters(constrains, verbose=True)
    scene.gen_sound_from_parameters(p, i, verbose=True)
    index += 1
    uf.print_bar(index,num_sounds)
'''
num_sounds2 = 30
index = 0
for i in range(num_sounds):
    constrains = sc.get_constrains(['only_available', 'particle', 'volume_hi', 'never_rev'])
    p = scene.gen_random_parameters(constrains)
    scene.gen_sound_from_parameters(p, num_sounds + i)
    index += 1
    uf.print_bar(index,num_sounds2)
'''


mix = scene.resolve_score_stereo(global_rev=True,fade_in=3000,fade_out=4000)
mix = scene.resolve_score_stereo(global_rev=False,fade_in=3000,fade_out=4000)
allocator.write_local(mix, '2')
scene.plot_score(dimensions=3)
