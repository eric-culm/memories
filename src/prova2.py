import matplotlib.pyplot as plt
import numpy as np
from modules import *
import random

scene = Scene(main_dur=40)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

sound1 = scene.get_random_sound('instrumental', 'buchla', 0, 60)

scene.gen_random_parameters()
scene.plot_score()
