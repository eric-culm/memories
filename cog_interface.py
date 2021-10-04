import os,sys
sys.path.append("source")
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
        '''setup func'''
        #nothing to load here

    @cog.input("type", type=str, default="episode", options=["episode","dream"], help="Type of file to generate. Episode is a single scenario, dream is a concatenation of episodes. For 'dream' only 'fast', 'max_num_sounds', 'length' and 'memories' have an effect")
    @cog.input("length", type=int, default=60, help="Soundfile length in minutes")
    @cog.input("fast", type=bool, default=True, help="If True disables the most resource-demanding processes to speed up generation")
    @cog.input("max_num_sounds", type=int, default=50, help="Maximum number of simultaneous sounds")
    @cog.input("memories_instrumental", type=str, default='all', help="Type of instrumental memories that can occur in the dream. List of strings. Options: 'all', 'africanPercs', 'ambient1', 'buchla', 'buchla2', 'classical', 'classical2', 'guitarAcoustic', 'guitarBaroque', 'jazz', 'organ', 'percsWar', 'percussions', 'pianoChill', 'pianoDreamy', 'pianoSmooth'")
    @cog.input("memories_soundscape", type=str, default='all', help="Type of soundscape memories that can occur in the dream. List of strings. Options: 'all', airport', 'birdsStreet', 'forest', 'library', 'mixed', 'office', 'rain', 'sea', 'train', 'wind'")
    @cog.input("density", type=float, default=0.2, help="Density of sound events [range 0-1]")
    @cog.input("score_diversity", type=float, default=0.2, help="Diversity of sound archetye choice [range 0-1]")
    @cog.input("timbre_diversity", type=float, default=0.2, help="Diversity of chosen timbres")
    @cog.input("carpet", type=bool, default=True, help="If True it is very probable that there is always a long sound at low volume")
    @cog.input("perc_particles", type=float, default=0.5, help="Probability of having fast and percussive sounds")
    @cog.input("enhance_random", type=bool, default=False, help="If true some unpredictable parameters are set to random")
    @cog.input("complete_random", type=bool, default=False, help="If true everything is random despite what is selected in the UI")
    @cog.input("global_rev_amount", type=float, default=0.1, help="Amount of global reverb [range 0-1]")
    @cog.input("global_stretch", type=float, default=0., help="Amount of time stretching. Negative to shorten, positive to increase duration")
    @cog.input("global_shift", type=float, default=0., help="Amount of pitch shifting. Negative to lower, positive to increase pitch")
    @cog.input("output_type", type=str, default='mp3', options=["wav", "mp3"], help="Wav or mp3 output")

    def predict(self, type, length, fast, max_num_sounds, memories_instrumental,
                memories_soundscape, output_type, density, score_diversity, timbre_diversity, carpet,
                    perc_particles, complete_random, enhance_random, global_rev_amount,
                    global_stretch, global_shift):
        """Compute dream!!"""
        #init paths and classes
        output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
        output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"
        self.build = BuildScene(max_dur=60*length, max_num_sounds=max_num_sounds)
        self.dream = Dream(scene_maxdur=20, max_num_sounds=max_num_sounds)
        self.post = Postprocessing()

        #create memories dict
        if memories_instrumental == 'all':
            memories_instrumental = sc.check_available_models()['instrumental']
        else:
            memories_instrumental = memories_instrumental.split(',')
        if memories_soundscape == 'all':
            memories_soundscape = sc.check_available_models()['fieldrec']
        else:
            memories_soundscape = memories_soundscape.split(',')
        choice_dict = {'instrumental': memories_instrumental, 'fieldrec': memories_soundscape}
        print ("Memories dict: ", choice_dict)

        #set global proccesing bools
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

        #compute scene if desired
        if type == 'episode':
            if not complete_random:
                mix, score = self.build.build(length=1,
                                  density=density,
                                  score_diversity=score_diversity,
                                  sel_diversity=timbre_diversity,
                                  single_model=False,
                                  fixed_category=False,
                                  fixed_model=False,
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
        #or compute dream
        else:
            mix = self.dream.random_dream(length, neuro_choice=choice_dict)

        #cut silences longer than 3 secs
        mix = self.post.cut_silence_multichannel(mix)

        #write output to file
        print ('Writing sounds to file')
        swapped_sound = np.swapaxes(mix,0,1)
        soundfile.write(str(output_path_wav), swapped_sound, 44100, format='WAV', subtype='PCM_16')

        #convert to mp3 if desired
        if output_type == "mp3":
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(output_path_wav),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(output_path_mp3),
                ],
            )
            return output_path_mp3
        else:
            return output_path_wav
