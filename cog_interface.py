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
        "memories",
        type=str,
        default="all",
        help="Type of sound memories that can occur in the dream (list of comma-divided items). Options: all, africanPercs, ambient1, buchla, buchla2, classical, classical2, guitarAcoustic, guitarBaroque, jazz, organ, percsWar, percussions, pianoChill, pianoDreamy, pianoSmooth, airport, birdsStreet, forest, library, mixed, office, rain, sea, train, wind",
    )
    @cog.input(
        "length",
        type=float,
        min=0.2,
        max=10,
        default=1,
        help="Approximate length of the dream (in minutes)",
    )
    @cog.input(
        "depth",
        type=float,
        default=0.5,
        min=0,
        max=1,
        help="Dream depth",
    )
    @cog.input(
        "unconsciousness",
        type=float,
        default=0.7,
        min=0,
        max=1,
        help="Unconsciousness level",
    )
    @cog.input(
        "output_type",
        type=str,
        default="mp3",
        options=["wav", "mp3"],
        help="Output format",
    )
    def predict(
        self,
        memories,
        length,
        depth,
        unconsciousness,
        output_type,
    ):
        """Compute dream"""
        cut_silence = True

        output_path_wav = Path(tempfile.mkdtemp()) / "output.wav"
        output_path_mp3 = Path(tempfile.mkdtemp()) / "output.mp3"

        max_num_sounds = np.interp(unconsciousness, [0.0, 1.0], [10, 100])
        max_segment_length = np.interp(depth, [0.0, 1.0], [120, 10])

        self.dream = Dream(
            max_num_sounds=max_num_sounds, scene_maxdur=max_segment_length
        )
        self.post = Postprocessing()

        memories = memories.replace(" ", "")

        if "all" in memories:
            choice_dict = sc.check_available_models()
        else:
            choice_dict = {"instrumental": [], "fieldrec": []}
            available_instrumental = sc.check_available_models()["instrumental"]
            available_fieldrec = sc.check_available_models()["fieldrec"]

            memories = memories.split(",")
            print(available_instrumental)
            print(available_fieldrec)

            for i in memories:
                print(i)
                if i in available_instrumental:
                    choice_dict["instrumental"].append(i)
                if i in available_fieldrec:
                    choice_dict["fieldrec"].append(i)

            if choice_dict["instrumental"] == []:
                choice_dict.pop("instrumental")
            if choice_dict["fieldrec"] == []:
                choice_dict.pop("fieldrec")

        print("Memories dict: ", choice_dict)

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
