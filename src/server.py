import threading
from pythonosc import dispatcher
from pythonosc import osc_server
import matplotlib.pyplot as plt
import re
from modules import *
import numpy as np
import sys
import time
import sounddevice as sd
import configparser
import loadconfig

#load config variables
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
DUR = cfg.getint('main', 'dur')
SR = cfg.getint('sampling', 'sr_target')
SR_PROCESSING = cfg.getint('main', 'sr_processing')
TOTAL_IN_CHANNELS = cfg.getint('main', 'total_in_channels')
SERVER_IP = cfg.get('osc', 'server_ip')
S2C_PORT = cfg.getint('osc', 's2c_port')
MODEL_WEIGHTS_PATH = cfg.get('main', 'model_weights_path')
MEMORY_LT_PATH = cfg.get('main', 'memory_lt_path')
MEMORY_ST_PATH = cfg.get('main', 'memory_st_path')
CLIENT_PATH = cfg.get('osc', 'client_path')
SAMPLERNN_CODE_PATH = cfg.get('samplernn', 'samplernn_code_path')
SAMPLERNN_ENV_PATH = cfg.get('samplernn', 'samplernn_env_path')
SAMPLERNN_SR = cfg.getint('samplernn', 'samplernn_sr')

#create modules instances
print ('Loading modules')
memory = Memory(memory_lt_path=MEMORY_LT_PATH, memory_st_path=MEMORY_ST_PATH)
allocator = Allocator(server_shared_path='../shared', sr=96000,
                    client_shared_path=os.path.join(CLIENT_PATH, 'shared'))
content_filter = FilterSound(memory_bag=memory.get_memory_lt(), threshold=0.0, random_prob=0.0, env_length=200)
dummy_model = DummyModel(dur=16384, latent_dim=100)

model_parameters = ['verbose=False', 'model_size=64', 'variational=True',
                    'latent_dim=100',]

VAE = VAE_model(architecture='WAVE_complete_net', weights_path=MODEL_WEIGHTS_PATH,
                parameters=model_parameters, device='cpu')
lop = LatentOperators(VAE.model.latent_dim)
#SRNN = SampleRNN(sr=SAMPLERNN_SR, code_path=SAMPLERNN_CODE_PATH, env_path=SAMPLERNN_ENV_PATH)



ins = {
        0: InputChannel(dur=DUR, channel=0, total_channels=TOTAL_IN_CHANNELS, sr=SR),
        1: InputChannel(dur=DUR, channel=1, total_channels=TOTAL_IN_CHANNELS, sr=SR)
    }

filters = {
        0: FilterStream(frequency=1, streaming_object=ins[0], filtering_object=content_filter),
        1: FilterStream(frequency=1, streaming_object=ins[1], filtering_object=content_filter)
        }

post = Postprocessing(SR_PROCESSING, '../IRs/revs/divided/')


'''
SRNN.build_train_string('cazzo', 'culo')
sys.exit(0)
'''

def rec(unused_addr, args, channel, flag):
    '''
    enable/disable input stream on one single channel
    args:
    1: target input channel
    2: 1=on, 0=off
    '''
    ins[channel].rec(flag)

def in_meter(unused_addr, args, channel, flag):
    '''
    enable/disable metering on one single channel
    args:
    1: target input channel
    2: 1=on, 0=off
    '''
    ins[channel].meter_continuous(flag)


def get_inamp(unused_addr, args, channel):
    '''
    output instant amplitude of one channel
    args:
    1: target input channe.
    '''
    ins[channel].meter_instantaneous()

def filter_input_sound(unused_addr, args, channel):
    buffer = ins[channel].get_buffer()
    filtered, sim = content_filter.filter_sound(buffer)
    return filtered

def collect_stream_stimuli(unused_addr, args, channel, flag, memory_type):
    filters[channel].filter_stream(flag, channel, memory, memory_type)

def get_memory_state(unused_addr, args):
    lt, st, rt = memory.get_state()
    print ('\nLong term memory len: ' + str(lt))
    print ('Short term memory len: ' + str(st))
    print ('Real time memory len: ' + str(rt))

def write_st_local(unused_addr, args, query_name):
    sounds = memory.get_memory_st()
    allocator.write_local(sounds, query_name)

def write_to_client(unused_addr, args, query_name):
    allocator.to_client(query_name)

def compute_quantization_grid(unused_addr, args, memory_type):
    if memory_type == 'lt':
        input = memory.get_memory_lt()
    if memory_type == 'st':
        input = memory.get_memory_st()
    VAE.compute_quantization_grid(input, memory_type)

def gen_random(unused_addr, args):
    z = VAE.decode_int(VAE.gen_random_z())
    allocator.write_local(z, 'random')
    allocator.to_client('random')

def gen_random_quant(unused_addr, args):
    z = VAE.decode_int(VAE.quantize(VAE.gen_random_z()))
    allocator.write_local(z, 'random')
    allocator.to_client('random')

def gen_random_blur(unused_addr, args):
    z = VAE.quantize(VAE.gen_random_z())
    out = [VAE.decode_int(z)]
    mul = 0.0
    for i in range(10):
        z1 = lop.blur(z, mul)
        #z1 = VAE.quantize(VAE.gen_random_z())
        x1 = VAE.decode_int(z1)
        out.append(x1)
        mul += 0.5
    allocator.write_local(out, 'random')
    allocator.to_client('random')

def gen_random_spike(unused_addr, args, n_spikes):
    z = VAE.quantize(VAE.gen_random_z())
    out = [VAE.decode_int(z)]
    for i in range(10):
        z1 = lop.spike(z, n_spikes)
        #z1 = VAE.quantize(VAE.gen_random_z())
        x1 = VAE.decode_int(z1)
        out.append(x1)
    allocator.write_local(out, 'random')
    allocator.to_client('random')

def gen_sequence(unused_addr, args, out_len, num_buffers, num_clusters, input_sound,
                max_len, sil_prob, sil_len, stretch_prob, stretch_len, cluster):
    print ('generating sequence')
    sil_prob = np.array(sil_prob.split(' '), dtype=np.float32)
    sil_len = np.array(sil_len.split(' '), dtype=np.float32)
    stretch_prob = np.array(stretch_prob.split(' '), dtype=np.float32)
    stretch_len = np.array(stretch_len.split(' '), dtype=np.float32)
    cluster = np.array(cluster.split(' '), dtype=np.float32)

    in_folder = '/Users/eric/Desktop/memories/input_sounds'
    file = os.path.join(in_folder, input_sound)
    max_len = max_len * SR_PROCESSING
    sounds = post.load_split(file, max_len)

    buffers = []
    if num_buffers > 1:
        for i in range(num_buffers):
            curr_buffer = post.concat_split(sounds, out_len, num_clusters, sil_prob, sil_len,
                                        stretch_prob, stretch_len, cluster)
            try:
                curr_buffer = post.reverb(curr_buffer, 'any')
                #pass
            except:
                pass
            buffers.append(curr_buffer)
    else:
        out = post.concat_split(sounds, out_len, num_clusters, sil_prob, sil_len,
                                    stretch_prob, stretch_len, cluster)

    out = post.distribute_pan_stereo(buffers)

    allocator.write_local(out, 'sequences')
    allocator.to_client('sequences')


    print ('sequence successfully generated')
















dispatcher = dispatcher.Dispatcher()
dispatcher.map("/print", print, 'text')
dispatcher.map("/rec", rec, 'channel', 'flag')
dispatcher.map("/in_meter", in_meter, 'channel', 'flag')
dispatcher.map("/get_inamp", get_inamp, 'channel')
dispatcher.map("/filter_input_sound", filter_input_sound, 'channel')
dispatcher.map("/collect_stream_stimuli", collect_stream_stimuli, 'channel', 'flag', 'memory_type')
dispatcher.map("/get_memory_state", get_memory_state, 'args')
dispatcher.map("/write_st_local", write_st_local, 'query_name')
dispatcher.map("/write_to_client", write_to_client, 'query_name')
dispatcher.map("/compute_quantization_grid", compute_quantization_grid, 'memory_type')
dispatcher.map("/gen_random", gen_random, 'args')
dispatcher.map("/gen_random_quant", gen_random_quant, 'args')
dispatcher.map("/gen_random_blur", gen_random_blur, 'args')
dispatcher.map("/gen_random_spike", gen_random_spike, 'n_spikes')
dispatcher.map("/gen_sequence", gen_sequence, 'out_len', 'num_buffers', 'num_clusters', 'input_sound', 'sil_prob',
                'sil_len', 'stretch_prob', 'stretch_len', 'cluster')



server = osc_server.ThreadingOSCUDPServer((SERVER_IP, S2C_PORT), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
'''
th = threading.Thread(target=server.serve_forever())
th.daemon = True
th.start()
'''
