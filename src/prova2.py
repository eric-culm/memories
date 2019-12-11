import numpy as np
import utility_functions as uf
import postprocessing_utils as pp
import matplotlib.pyplot as plt
import librosa
import scipy
from modules import *



#load config variables
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
DUR = cfg.getint('main', 'dur')
SR = cfg.getint('sampling', 'sr_target')
TOTAL_IN_CHANNELS = cfg.getint('main', 'total_in_channels')
SERVER_IP = cfg.get('osc', 'server_ip')
S2C_PORT = cfg.getint('osc', 's2c_port')
MODEL_WEIGHTS_PATH = cfg.get('main', 'model_weights_path')
MEMORY_LT_PATH = cfg.get('main', 'memory_lt_path')
MEMORY_ST_PATH = cfg.get('main', 'memory_st_path')
CLIENT_PATH = cfg.get('osc', 'client_path')

#create modules instances
print ('Loading modules')
memory = Memory(memory_lt_path=MEMORY_LT_PATH, memory_st_path=MEMORY_ST_PATH)
allocator = Allocator(server_shared_path='../shared',
                    client_shared_path=os.path.join(CLIENT_PATH, 'shared'))
content_filter = FilterSound(memory_bag=memory.get_memory_lt(), threshold=0.0, random_prob=0.0, env_length=200)
dummy_model = DummyModel(dur=16384, latent_dim=100)

model_parameters = ['verbose=False', 'model_size=64', 'variational=True',
                    'latent_dim=100',]

VAE = VAE_model(architecture='WAVE_complete_net', weights_path=MODEL_WEIGHTS_PATH,
                parameters=model_parameters, device='cpu')

#lop = LatentOperators(VAE.model.latent_dim)
lop = LatentOperators(100
)

post = Postprocessing(16000, '../IRs/revs/divided/')



num_sounds = 150
sounds = []

print ('computing vae')
for i in range(num_sounds):
    s = VAE.decode_int(VAE.quantize(VAE.gen_random_z()))
    sounds.append(s)

'''
s1 = '/home/eric/Downloads/voice_prova.wav'


out_file = '/home/eric/Desktop/quick_sounds/CULISMO'
dur = 16000
sounds = []
samples, sr = librosa.core.load(s1, sr=16000)
seg = np.arange(0, len(samples), dur)

for i in seg:
    try:
        sounds.append(samples[i:i+dur])
    except:
        pass

print (len(sounds))
'''



print ('\nbuilding sound')

#curves
sil_prob_curve =(1 - np.arange(100) / 100 )* 0.65
sil_len_curve = 1 - (np.arange(100) / 200 + 0.5)

stretch_prob_curve = (np.arange(100) / 100)
stretch_factor_curve = (1 - np.arange(100)) / 100
stretch_factor_curve = np.interp(stretch_factor_curve, (0, 1), (0., 1.5))

#sil_prob_curve = np.ones(100) * 0.
#sil_len_curve = np.ones(100) * 0.

#stretch_prob_curve = np.ones(100)
#stretch_factor_curve = np.ones(100) * 0.4

num_buffers = 8
out = []
for i in range(num_buffers):
    curr_buffer = post.concat_split(sounds, 60, sil_prob_curve, sil_len_curve, stretch_prob_curve, stretch_factor_curve, 150)
    try:
        curr_buffer = post.reverb(curr_buffer, 'any')
    except:
        pass
    out.append(curr_buffer)


out = post.distribute_pan_stereo(out)
librosa.output.write_wav('/home/eric/Desktop/quick_sounds/CULISMO.wav', out, 16000)
#uf.wavwrite(out, 16000, '/home/eric/Desktop/quick_sounds/CULISMO.wav')
#allocator.write_local(buffer, 'random')
#allocator.to_client('random')
print ('\nculo')
#print (len(buffer))
#plt.plot(out)
#plt.show()
