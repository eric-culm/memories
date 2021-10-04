
from modules import *
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

GRID_SAVE = cfg.get('vae', 'vae_quantization_grid')
SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')

VAE = VAE(architecture='WAVE_complete_net', weights_path='../vae_data/models/init_before_convergence',
                device='cpu')

quantization_grid = VAE.compute_quantization_grid(SRNN_DATA_PATH, GRID_SAVE)
