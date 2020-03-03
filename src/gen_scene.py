import matplotlib.pyplot as plt
import numpy as np
from modules import *
import scene_constrains as sc
import random
import utility_functions as uf

print (sc.check_available_models())

scene = Scene(main_dur=60)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

choice_dict = {'instrumental': ['percsWar']}
dream = Dream()
scene.gen_macro()
build = BuildScene()
a = dream.gen_durations(60*60, 60)
print ('cazzo', len(a))
rand = True
if not rand:
    mix, score = build.build(length=1,
                      density=0.2,
                      score_diversity=0.4,
                      sel_diversity=0.2,
                      single_model=False,
                      fixed_category=False,
                      fixed_model=False,
                      neuro_choice=choice_dict,
                      fast=True,
                      carpet=True,
                      perc_particles=0.,
                      enhance_random=False,
                      complete_random=True,
                      global_rev=False,
                      global_rev_amount=0.5,
                      global_stretch_dir=0,
                      global_stretch=0,
                      global_shift_dir=0,
                      global_shift=0,
                      verbose=False)

else:
    mix, score, p = build.random_build(length=1, neuro_choice=choice_dict)
mix =post.cut_silence_multichannel(mix)
print ('sfigatos')
print (mix.shape)
allocator.write_local(mix, 'coglionazzo')
scene.load_score(score)
#scene.plot_score(dimensions=3)
