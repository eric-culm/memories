import threading
from pythonosc import dispatcher
from pythonosc import osc_server
from modules import *
import numpy as np
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
MEMORY_BAG = cfg.get('main', 'memory_bag')
SERVER_IP = cfg.get('osc', 'server_ip')
S2C_PORT = cfg.getint('osc', 's2c_port')

#create modules instances
memory_bag = np.load(MEMORY_BAG)
content_filter = FilterSound(memory_bag=memory_bag, threshold=0.0, random_prob=0.0, env_length=200)
dummy_model = DummyModel(dur=16384, latent_dim=100)


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

def filter_input_stream(unused_addr, args, channel, flag):
    filters[channel].filter_stream(flag)
    if flag == 0:
        bag = filters[channel].get_bag()
        print (len(bag))









dispatcher = dispatcher.Dispatcher()
dispatcher.map("/rec", rec, 'channel', 'flag')
dispatcher.map("/in_meter", in_meter, 'channel', 'flag')
dispatcher.map("/get_inamp", get_inamp, 'channel')
dispatcher.map("/filter_input_sound", filter_input_sound, 'channel')
dispatcher.map("/filter_input_stream", filter_input_stream, 'channel', 'flag')

server = osc_server.ThreadingOSCUDPServer((SERVER_IP, S2C_PORT), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
'''
th = threading.Thread(target=server.serve_forever())
th.daemon = True
th.start()
'''
