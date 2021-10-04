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

choice_dict = {'instrumental': ['pianoDreamy']}
dream = Dream()
#scene.gen_macro()
build = BuildScene()
#a = dream.gen_durations(60*60, 60)
#print ('cazzo', len(a))
rand = False
if not rand:
    mix, score = build.build(length=1,
                      density=0.2,
                      score_diversity=0.2,
                      sel_diversity=0.3,
                      single_model=True,
                      fixed_category='fieldrec',
                      fixed_model='forest',
                      neuro_choice=False,
                      fast=True,
                      carpet=True,
                      perc_particles=0.5,
                      enhance_random=False,
                      complete_random=False,
                      global_rev=True,
                      global_rev_amount=0.1,
                      global_stretch_dir=0,
                      global_stretch=0,
                      global_shift_dir=0,
                      global_shift=0.1,
                      verbose=False)

else:
    mix, score, p = build.random_build(length=1, neuro_choice=choice_dict)
mix =post.cut_silence_multichannel(mix)
print ('sfigatos')
print (mix.shape)
allocator.write_local(mix, 'stramerda2')
scene.load_score(score)
#scene.plot_score(dimensions=3)
