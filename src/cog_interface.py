import os
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
        """Load the model"""

    # @cog.input("input", type=Path, help="Input image path")
    #@cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, seed):
        """Compute prediction"""
        print ("coglione")
