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

choice_dict = {'instrumental': ['pianoDreamy', 'pianoChill', 'pianoSmooth']}

scene.gen_macro()
build = BuildScene()
mix, score = build.build(length=0.6,
                  density=0.2,
                  score_diversity=0.4,
                  sel_diversity=0.8,
                  single_model=False,
                  fixed_category='instrumental',
                  fixed_model='guitarAcoustic',
                  neuro_choice=choice_dict,
                  fast=True,
                  carpet=True,
                  perc_particles=1,
                  enhance_random=False,
                  complete_random=False,
                  global_rev=False,
                  global_rev_amount=0.5,
                  global_stretch_dir=0,
                  global_stretch=0,
                  global_shift_dir=0,
                  global_shift=0,
                  verbose=False)

allocator.write_local(mix, 'coglionazzo')
scene.load_score(score)
#scene.plot_score(dimensions=3)
