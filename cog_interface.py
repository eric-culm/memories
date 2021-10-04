import os,sys
sys.path.append("source")
#os.chdir("source")
import tempfile
from pathlib import Path

import cog
import matplotlib.pyplot as plt
import numpy as np
from modules import *
import scene_constrains as sc
import random
import utility_functions as uf


class GenDream(cog.Predictor):
    def setup(self):
        """Load the modules"""
        max_dur = 10
        self.scene = Scene(main_dur = 60 * max_dur)
        self.allocator = Allocator(server_shared_path='test', sr=44100,
                            client_shared_path='test')
        self.post = Postprocessing()
        #scene.gen_macro()

    @cog.input("type", type=str, default="dream", options=["episode","dream"], help="Type of file to generate. Episode is a single scenario, dream is a concatenation of episodes. For 'dream' only 'fast', 'max_num_sounds', 'length' and 'memories' have an effect")
    @cog.input("length", type=int, default=600, help="Soundfile length in minutes")
    @cog.input("fast", type=bool, default=True, help="If True disables most pitch shifting events to speed up generation")
    @cog.input("max_num_sounds", type=int, default=50, help="Maximum number of simultaneous sounds")
    @cog.input("memories", type=list, default=['all'], help="Type of memories that can occur in the dream. List of strings. Options:")

    @cog.input("density", type=float, default=0.2, help="Density of sound events [range 0-1]")
    @cog.input("score_diversity", type=float, default=0.2, help="Diversity of sound archetye choice [range 0-1]")
    @cog.input("timbre_diversity", type=float, default=0.2, help="Diversity of chosen timbres")
    @cog.input("carpet", type=bool, default=True, help="If True it is very probable that there is always a long sound at low volume")
    @cog.input("perc_particles", type=float, default=0.5, help="Probability of having fast and percussive sounds")
    @cog.input("enhance_random", type=bool, default=False, help="If true some unpredictable parameters are set to random")
    @cog.input("complete_random", type=bool, default=False, help="If true everything is random despite what is selected in the UI")
    @cog.input("global_rev_amount", type=float, default=0.1, help="Amount of global reverb [range 0-1]")
    @cog.input("global_stretch", type=float, default=0., help="Amount of time stretching. Negative to shorten, positive to increase duration")
    @cog.input("global_shift", type=float, default=-2000., help="Amount of pitch shifting. Negative to lower, positive to increase pitch")

    def predict(self, type, length, max_num_sounds, density, score_diversity, timbre_diversity, fast, carpet,
                    perc_particles, complete_random, enhance_random, global_rev_amount,
                    global_stretch, global_shift):
        """Compute dream!!"""
        self.build = BuildScene(max_dur=60*length, max_num_sounds=max_num_sounds)
        self.dream = Dream(scene_maxdur=20, max_num_sounds=max_num_sounds)

        choice_dict = {'instrumental': ['pianoDreamy']}

        if global_rev_amount == 0.0:
            global_rev = False
        else:
            global_rev = True

        if global_stretch >= 0.:
            global_stretch_dir = 1
        else:
            global_stretch_dir = 0

        if global_shift >= 0.:
            global_shift_dir = 1
        else:
            global_shift_dir = 0

        if type == 'episode':
            if not complete_random:
                mix, score = self.build.build(length=1,
                                  density=density,
                                  score_diversity=score_diversity,
                                  sel_diversity=timbre_diversity,
                                  single_model=False,
                                  fixed_category='fieldrec',
                                  fixed_model='forest',
                                  neuro_choice=choice_dict,
                                  fast=fast,
                                  carpet=carpet,
                                  perc_particles=perc_particles,
                                  enhance_random=enhance_random,
                                  complete_random=False,
                                  global_rev=global_rev,
                                  global_rev_amount=global_rev_amount,
                                  global_stretch_dir=global_stretch_dir,
                                  global_stretch=0,
                                  global_shift_dir=global_shift_dir,
                                  global_shift=global_shift,
                                  verbose=False)

            else:
                mix, score, p = self.build.random_build(length=1, neuro_choice=choice_dict)
        else:
            mix = dream.random_dream(length, neuro_choice=choice_dict)

        mix = self.post.cut_silence_multichannel(mix)
        self.allocator.write_local(mix, 'test')
        self.scene.load_score(score)
        #self.scene.plot_score(dimensions=3)
