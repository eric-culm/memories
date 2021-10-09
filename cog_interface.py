import os
import sys

sys.path.append("source")
import random
import subprocess
import tempfile
import warnings
from pathlib import Path

import cog
import matplotlib.pyplot as plt
import numpy as np
import scene_constrains as sc
import utility_functions as uf
from ir_analysis import analyze_irs
from modules import *
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning)


class GenDream(cog.Predictor):
    def setup(self):
        """setup func"""
        print("creating IR analysis file")
        analyze_irs()

    @cog.input(
        "type",
        type=str,
        default="episode",
        options=["episode", "dream"],
        help="Type of file to generate. Episode is a single scenario, dream is a concatenation of episodes. For 'dream' only 'fast', 'max_num_sounds', 'dream_length', 'max_episode_length', 'memories', parameters have an effect",
    )
    @cog.input(
        "memories",
        type=str,
        default="all",
        help="Type of sound memories that can occur in the dream (list of comme-divided items). Options: all, africanPercs, ambient1, buchla, buchla2, classical, classical2, guitarAcoustic, guitarBaroque, jazz, organ, percsWar, percussions, pianoChill, pianoDreamy, pianoSmooth, airport, birdsStreet, forest, library, mixed, office, rain, sea, train, wind",
    )
    @cog.input(
        "dream_length",
        type=float,
        default=3,
        help="Approximative length of soundfile to generate in minutes (only for dream type)",
    )
    @cog.input(
        "max_episode_length",
        type=int,
        default=60,
        min=10,
        max=60,
        help="Maximum episodes duration in seconds. With lower values dreams will contain more episodes",
    )
    @cog.input(
        "max_num_sounds",
        type=int,
        default=50,
        min=0,
        max=300,
        help="Maximum number of simultaneous sounds. Higher values for more dense outputs",
    )
    @cog.input(
        "density",
        type=float,
        default=0.8,
        min=0,
        max=1,
        help="Density of sound events [range 0-1]",
    )
    @cog.input(
        "score_diversity",
        type=float,
        default=0.6,
        min=0,
        max=1,
        help="Diversity of sound archetypes choice [range 0-1]",
    )
    @cog.input(
        "timbre_diversity",
        type=float,
        default=0.6,
        min=0,
        max=1,
        help="Diversity of chosen timbres",
    )
    @cog.input(
        "perc_particles",
        type=float,
        default=0.5,
        min=0,
        max=1,
        help="Probability of having fast and percussive sounds",
    )
    @cog.input(
        "global_rev_amount",
        type=float,
        default=0.1,
        min=0,
        max=1,
        help="Amount of global reverb [range 0-1]",
    )
    @cog.input(
        "global_stretch",
        type=float,
        default=0,
        help="Amount of global time stretching. Negative to shorten, positive to lengthen duration",
    )
    @cog.input(
        "global_shift",
        type=float,
        default=0,
        help="Amount of global pitch shifting. Negative to decrease, positive to increase pitch",
    )
    @cog.input(
        "output_type",
        type=str,
        default="wav",
        options=["wav", "mp3"],
        help="Wav or mp3 output",
    )
    @cog.input(
        "enhance_random",
        type=bool,
        default=False,
        help="If true some unpredictable parameters are set to random",
    )
    @cog.input(
        "complete_random",
        type=bool,
        default=False,
        help="If true everything is random despite what is selected in the UI",
    )
    @cog.input(
        "carpet",
        type=bool,
        default=True,
        help="If True it is very probable that there is always a long sound at low volume",
    )
    @cog.input(
        "fast",
        type=bool,
        default=True,
        help="If True disables the most resource-demanding processes to speed up generation",
    )
    @cog.input(
        "cut_silence",
        type=bool,
        default=True,
        help="Cut all silences longer than 5 seconds",
    )
    def predict(
        self,
        type,
        dream_length,
        max_episode_length,
        fast,
        max_num_sounds,
        memories,
        output_type,
        density,
        score_diversity,
        timbre_diversity,
        carpet,
        perc_particles,
        complete_random,
        enhance_random,
        global_rev_amount,
        global_stretch,
        global_shift,
        cut_silence,
    ):
        """Compute dream"""
        # init paths and classes
        output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
        output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"
        self.build = BuildScene(
            max_dur=max_episode_length, max_num_sounds=max_num_sounds
        )
        self.dream = Dream(
            max_num_sounds=max_num_sounds, scene_maxdur=max_episode_length
        )
        self.post = Postprocessing()

        memories = memories.replace(" ", "")

        if 'all' in memories:
            choice_dict = sc.check_available_models()
        else:
            choice_dict = {"instrumental": [], "fieldrec": []}
            available_instrumental = sc.check_available_models()["instrumental"]
            available_fieldrec = sc.check_available_models()["fieldrec"]

            memories = memories.split(",")
            print (available_instrumental)
            print (available_fieldrec)

            for i in memories:
                print (i)
                if i in available_instrumental:
                    choice_dict["instrumental"].append(i)
                if i in available_fieldrec:
                    choice_dict["fieldrec"].append(i)

            if choice_dict["instrumental"] == []:
                choice_dict.pop("instrumental")
            if choice_dict["fieldrec"] == []:
                choice_dict.pop("fieldrec")

        print("Memories dict: ", choice_dict)

        # set global proccesing bools
        if global_rev_amount == 0.0:
            global_rev = False
        else:
            global_rev = True

        if global_stretch >= 0.0:
            global_stretch_dir = 1
        else:
            global_stretch_dir = 0

        if global_shift >= 0.0:
            global_shift_dir = 1
        else:
            global_shift_dir = 0

        # compute scene if desired
        if type == "episode":
            if not complete_random:
                mix, score = self.build.build(
                    length=1,
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
                    verbose=False,
                )

            else:
                mix, score, p = self.build.random_build(
                    length=1, neuro_choice=choice_dict
                )
        # or compute dream
        else:
            mix = self.dream.random_dream(dream_length * 60, neuro_choice=choice_dict)

        # cut silences longer than 3 secs
        if cut_silence:
            mix = self.post.cut_silence_multichannel(mix)

        # write output to file
        print("Writing sounds to file")
        swapped_sound = np.swapaxes(mix, 0, 1)
        soundfile.write(
            str(output_path_wav), swapped_sound, 44100, format="WAV", subtype="PCM_16"
        )

        # convert to mp3 if desired
        if output_type == "mp3":

            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(output_path_wav),
                    "-ab",
                    "320k",
                    str(output_path_mp3),
                ],
            )

            # AudioSegment.from_wav(output_path_wav).export(output_path_mp3, format="mp3", bitrate="320k")
            return output_path_mp3
        else:
            return output_path_wav
