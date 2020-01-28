import matplotlib.pyplot as plt
import numpy as np
from modules import *
import random

scene = Scene(main_dur=40)
allocator = Allocator(server_shared_path='../shared', sr=44100,
                    client_shared_path='cazzo')
post = Postprocessing()

sound1 = scene.get_random_sound('instrumental', 'buchla', 0, 60)

def culo(input, d=0):
    return input + d

def cazzo(input):
    return cazzo +10



funcs = {'culo' : culo(0, d=4),
            'cazzo': cazzo}

a = funcs['culo'](4)


print (str(a))
