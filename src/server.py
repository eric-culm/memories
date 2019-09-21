"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math
import threading
from pythonosc import dispatcher
from pythonosc import osc_server
from modules import *
import numpy as np
import time
import sounddevice as sd

dur = 16384
memory_bag = np.load('/home/eric/Desktop/memories/dataset/matrices/digits_validation_predictors_fold_0.npy')


rec_buffer_0 = np.zeros(dur)
filtered_sounds = []
stream_filter = FilterStream(memory_bag=memory_bag, threshold=0., random_prob=0.0, env_length=200)


class InputChannel:
    def __init__(self, dur, channel):
        self.dur = dur
        self.channel = channel
        self.buffer = np.zeros(dur)
        self.stream =  sd.InputStream(channels=1, blocksize=512 , callback=self.rec_callback)

    def rec_callback(self, indata, frames, time, status):
        '''
        record sliding buffers of length DUR, updating every bloch size
        '''
        if status:
            print(status)
        self.buffer = np.roll(self.buffer, -frames , axis=0)  #shift vector
        self.buffer[-frames:] = indata[:,self.channel] #add new data

    def rec(self, flag):
        if flag == 1:
            print ("")
            print ("Audio stream started")
            print ("")
            self.stream.start()
        if flag == 0:
            print ("")
            print ("Audio stream closed")
            print ("")
            self.stream.stop()

    def get_buffer():
        return self.buffer

    def meter_continuous(self, flag):
        self.meterflag = flag
        while self.meterflag == 1:
            peak = max(abs(self.buffer))
            print_peak = str(np.round(peak, decimals=3))
            meter_string =  "IN " + str(self.channel) + ": " + print_peak
            print ('\r', meter_string, end='')
            #print ('\r', '', end='')
            time.sleep(0.05)

    def meter_instantaneous(self):
        peak = max(abs(self.buffer))
        print_peak = str(np.round(peak, decimals=3))
        meter_string =  "IN " + str(self.channel) + ": " + print_peak
        print ('\r', meter_string, end='')





'''

def meter(unused_addr, args, flag):
    global rec_buffer_0

    peak = max(abs(rec_buffer_0))

    print (peak)
'''
rec1 = InputChannel(16000, 0)

def rec(unused_addr, args, flag):

    rec1.rec(flag)

def meter(unused_addr, args, flag):

    rec1.meter_instantaneous()




dispatcher = dispatcher.Dispatcher()
dispatcher.map("/rec", rec, 'time_rec')
dispatcher.map("/meter", meter, 'flag')
#dispatcher.map("/culo", culo, 'flag')

server = osc_server.ThreadingOSCUDPServer(('127.0.0.1', 5005), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
'''
th = threading.Thread(target=server.serve_forever())
th.daemon = True
th.start()
'''
