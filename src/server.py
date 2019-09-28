import threading
from pythonosc import dispatcher
from pythonosc import osc_server
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
TOTAL_IN_CHANNELS = cfg.getint('main', 'total_in_channels')
SERVER_IP = cfg.get('osc', 'server_ip')
S2C_PORT = cfg.getint('osc', 's2c_port')
MODEL_WEIGHTS_PATH = cfg.get('main', 'model_weights_path')
MEMORY_LT_PATH = cfg.get('main', 'memory_lt_path')
MEMORY_ST_PATH = cfg.get('main', 'memory_st_path')
CLIENT_PATH = cfg.get('osc', 'client_path')

#create modules instances
memory = Memory(memory_lt_path=MEMORY_LT_PATH, memory_st_path=MEMORY_ST_PATH)
allocator = Allocator(server_shared_path='../shared',
                    client_shared_path=os.path.join(CLIENT_PATH, 'shared'))
content_filter = FilterSound(memory_bag=memory.get_memory_lt(), threshold=0.0, random_prob=0.0, env_length=200)
dummy_model = DummyModel(dur=16384, latent_dim=100)

model_parameters = ['verbose=False', 'model_size=64', 'variational=True',
                    'latent_dim=100',]

VAE = VAE_model(architecture='WAVE_CNN_complete_net', weights_path=MODEL_WEIGHTS_PATH,
                parameters=model_parameters, device='cpu')



ins = {
        0: InputChannel(dur=DUR, channel=0, total_channels=TOTAL_IN_CHANNELS),
        1: InputChannel(dur=DUR, channel=1, total_channels=TOTAL_IN_CHANNELS)
    }

filters = {
        0: FilterStream(frequency=1, streaming_object=ins[0], filtering_object=content_filter),
        1: FilterStream(frequency=1, streaming_object=ins[1], filtering_object=content_filter)
        }



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

def collect_stream_stimuli(unused_addr, args, channel, flag):
    filters[channel].filter_stream(flag, channel, memory)

def get_memory_state(unused_addr, args):
    lt, st = memory.get_state()
    print ('Long term memory len: ' + str(lt))
    print ('Short term memory len: ' + str(st))

def write_st_local(unused_addr, args, query_name):
    sounds = memory.get_memory_st()
    allocator.write_local(sounds, query_name)

def write_to_client(unused_addr, args, query_name):
    allocator.to_client(query_name)









dispatcher = dispatcher.Dispatcher()
dispatcher.map("/print", print, 'text')
dispatcher.map("/rec", rec, 'channel', 'flag')
dispatcher.map("/in_meter", in_meter, 'channel', 'flag')
dispatcher.map("/get_inamp", get_inamp, 'channel')
dispatcher.map("/filter_input_sound", filter_input_sound, 'channel')
dispatcher.map("/collect_stream_stimuli", collect_stream_stimuli, 'channel', 'flag')
dispatcher.map("/get_memory_state", get_memory_state, 'args')
dispatcher.map("/write_st_local", write_st_local, 'query_name')
dispatcher.map("/write_to_client", write_to_client, 'query_name')


server = osc_server.ThreadingOSCUDPServer((SERVER_IP, S2C_PORT), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
'''
th = threading.Thread(target=server.serve_forever())
th.daemon = True
th.start()
'''
