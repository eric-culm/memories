import matplotlib.pyplot as plt
import numpy as np
from modules import *
import scene_constrains as sc
import random

scene = Scene(main_dur=40)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

constrains = only_available
constrains = sc.get_constrains(['only_available', 'prefer_hq'])
print(constrains)
p = scene.gen_random_parameters(constrains)
scene.gen_sound_from_parameters(p, 0)
#scene.plot_score()
