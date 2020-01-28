import matplotlib.pyplot as plt
import numpy as np
from modules import *

scene = Scene(main_dur=40)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')

sound1 = scene.get_random_sound('instrumental', 'buchla', 0, 60)
sound2 = scene.get_random_sound('instrumental', 'buchla', 1, 60)
sound3 = scene.get_random_sound('instrumental', 'buchla', 2, 60)
sound4 = scene.get_random_sound('instrumental', 'classical', 0, 60)
sound5 = scene.get_random_sound('instrumental', 'classical', 0, 60)

scene.gen_scored_sound(sound1, 30, 0.7, 0.7, pan=[0,0.2], id=0, fade_in=4000, fade_out=5000, segment=True)
scene.gen_scored_sound(sound2, 30, .8, 0.6, pan=[-1,-1], id=1, fade_in=4000, fade_out=5000, segment=True)
scene.gen_scored_sound(sound3, 20, 0.8, 0.3, pan=[0.9,0.7], id=2, fade_in=4000, fade_out=5000, segment=True)
scene.gen_scored_sound(sound4, 10, 0.6, 0.23, pan=[-.2,-.2], id=3, rev=True, stretch=0.4, fade_in=4000, fade_out=5000, segment=True)
scene.gen_scored_sound(sound5, 40, 0.2, 0., pan=[-.7,-.2], id=4, rev=True, fade_in=4000, fade_out=5000, shift=-36, segment=True)


mix = scene.resolve_score_stereo(0,4000)
allocator.write_local(mix, '2')
scene.plot_score(dimensions=3)
