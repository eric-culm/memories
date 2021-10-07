import os, sys
import loadconfig
import configparser
from pydub import AudioSegment
import multiprocessing
import utility_functions as uf
import random

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')

folders_macro = ['instrumental', 'fieldrec']
folders_macro_wav = [os.path.join(SRNN_DATA_PATH, i) for i in folders_macro]


a = 'srnn_data/fieldrec/birdsStreet/sounds/dur_30/model_1/ep01-02-2020_05:10:50-s37.wav'

def make_mp3_dirs(input_dir):
    target_dir = input_dir.split('/')
    target_dir[0] = target_dir[0] + '_mp3'
    target_dir = '/'.join(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def get_mp3_name(input_dir):
    target_dir = input_dir.split('/')
    target_dir[0] = target_dir[0] + '_mp3'
    target_dir = '/'.join(target_dir)
    target_dir = target_dir[:-3] + 'mp3'

    return target_dir

def convert_to_mp3(wavname, mp3name):
    print (wavname)
    AudioSegment.from_wav(wavname).export(mp3name, format="mp3", bitrate="320k")



pool = multiprocessing.Pool(processes=32)
count = 0
for i in folders_macro_wav:
    contents = os.listdir(i)
    contents = [s for s in contents if '.DS_Store' not in s]
    random.shuffle(contents)
    for c in contents:
        c_wav = os.path.join(i, c, "sounds")
        contents_2 = os.listdir(c_wav)
        contents_2 = [i for i in contents_2 if '.DS_Store' not in i]
        contents_2 = [os.path.join(c_wav, i) for i in contents_2]
        random.shuffle(contents_2)
        for m in contents_2:
            contents_3 = os.listdir(m)
            contents_3 = [i for i in contents_3 if '.DS_Store' not in i]
            contents_3 = [os.path.join(m, i) for i in contents_3]
            random.shuffle(contents_3)
            for i_m in contents_3:
                make_mp3_dirs(i_m)
            for s in contents_3:
                contents_4 = os.listdir(s)
                contents_4 = [i for i in contents_4 if '.DS_Store' not in i]
                contents_4 = [os.path.join(s, i) for i in contents_4]
                random.shuffle(contents_4)
                for soundfile in contents_4:
                    soundfile_mp3 = get_mp3_name(soundfile)
                    if not os.path.exists(soundfile_mp3):
                        count += 1
                        #pool.apply_async(convert_to_mp3, (soundfile,soundfile_mp3,), callback=callback_append)
                        #AudioSegment.from_wav(soundfile).export(soundfile_mp3, format="mp3", bitrate="320k")
print ("To process: ", count)
random.seed(412)
new_count = 0
for i in folders_macro_wav:
    contents = os.listdir(i)
    contents = [s for s in contents if '.DS_Store' not in s]
    random.shuffle(contents)
    for c in contents:
        c_wav = os.path.join(i, c, "sounds")
        contents_2 = os.listdir(c_wav)
        contents_2 = [i for i in contents_2 if '.DS_Store' not in i]
        contents_2 = [os.path.join(c_wav, i) for i in contents_2]
        random.shuffle(contents_2)
        for m in contents_2:
            contents_3 = os.listdir(m)
            contents_3 = [i for i in contents_3 if '.DS_Store' not in i]
            contents_3 = [os.path.join(m, i) for i in contents_3]
            random.shuffle(contents_3)
            for i_m in contents_3:
                make_mp3_dirs(i_m)
            for s in contents_3:
                contents_4 = os.listdir(s)
                contents_4 = [i for i in contents_4 if '.DS_Store' not in i]
                contents_4 = [os.path.join(s, i) for i in contents_4]
                random.shuffle(contents_4)
                for soundfile in contents_4:
                    soundfile_mp3 = get_mp3_name(soundfile)
                    if not os.path.exists(soundfile_mp3):
                        #AudioSegment.from_wav(soundfile).export(soundfile_mp3, format="mp3", bitrate="320k")
                        pool.starmap(convert_to_mp3, [(soundfile,soundfile_mp3)])
                        new_count += 1
                        uf.print_bar(new_count, count)

pool.close()
