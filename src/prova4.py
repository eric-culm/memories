import sounddevice as sd
import OSC
import os
import sys
import threading
import time
import utility_functions as uf
from keras.models import load_model
import use_conv_model_regression as conv
from scipy.stats import pearsonr
#import dataset_builders as db
import use_markovchain as mk
import numpy as np
import tensorflow as tf
import loadconfig
import random
import ConfigParser
from multiprocessing import Process
##
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

receive_address = '127.0.0.1', 7000
send_address = '127.0.0.1', 7001
server = OSC.OSCServer(receive_address)
client = OSC.OSCClient()
client.connect(send_address)

time.sleep(2)
print("OSC Server started")

global rec_buffer
global wait_status   #variable to check if the symth datapoind has been generated and classified
global temp_datapoint_settings
global temp_datapoint_verdict
global bar
global corr_buffer1
global corr_buffer2


rec_buffer_1 = np.zeros(16000)  #init circular recording buffers

def rec_callback(indata, outdata, frames, time, status):
    '''
    record circular buffers of length DUR, updating every bloch size
    '''
    global rec_buffer_1


    rec_buffer_1 = np.roll(rec_buffer_1, -512, axis=0)  #shift vector
    rec_buffer_1[-frames:] = indata[:,0]  #add new data



def bar(status, channel):
    tot_range = 50
    status = status * tot_range/10

    print ("(_|_)" + "="*status + "D" + " "*(tot_range-(tot_range/10)-status) + "(_|_)" + " Channel " + str(channel))

def meter(addr, tags, data, client_address):
    '''
    simple audio meter of a selected buffer
    '''
    global rec_buffer_1

    rec_buffer = rec_buffer_1
    peak = max(abs(rec_buffer))
    print (peak)
    max_range = 10
    peak = int((peak**0.3)*max_range)
    print ("|CH" + str(data[0]) + "|" + "|""="*peak*3 + "=D")




server.addMsgHandler('meter', meter)
server.addMsgHandler('/quit', quit_handler)



th = threading.Thread(target = server.serve_forever)
th.daemon = True  #automatically kills thread when quitting server
th.start()

if __name__ == '__main__':
    with sd.Stream(channels=AUDIO_IN_CHANNELS, blocksize=BLOCKSIZE, callback=rec_callback):
        print ("")
        print ("Audio stream started")  #audio recording stream
        input()
