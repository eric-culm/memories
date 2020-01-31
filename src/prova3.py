import matplotlib.pyplot as plt
import numpy as np
from modules import *
import scene_constrains as sc
import random
import utility_functions as uf

scene = Scene(main_dur=20)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

#constrains = only_available

num_sounds = 5
index = 0
for i in range(num_sounds):
    constrains = sc.get_constrains(['only_available', 'volume_mid', 'only_long_selected', 'very_long_scored', 'no_shift', 'stretch_long'])
    p = scene.gen_random_parameters(constrains)
    scene.gen_sound_from_parameters(p, i)
    index += 1
    uf.print_bar(index,num_sounds)

num_sounds2 = 30
index = 0
for i in range(num_sounds):
    constrains = sc.get_constrains(['only_available', 'particle', 'volume_hi', 'never_rev'])
    p = scene.gen_random_parameters(constrains)
    scene.gen_sound_from_parameters(p, num_sounds + i)
    index += 1
    uf.print_bar(index,num_sounds2)


mix = scene.resolve_score_stereo(global_rev=True,fade_in=3000,fade_out=4000)
mix = scene.resolve_score_stereo(global_rev=False,fade_in=3000,fade_out=4000)
allocator.write_local(mix, '2')
scene.plot_score(dimensions=3)
