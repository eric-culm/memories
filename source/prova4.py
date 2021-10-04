import numpy as np
import subprocess
import utility_functions as uf
import sys
from shutil import copyfile

# ffmpeg -i input.wav -vn -ar 44100 -ac 2 -b:a 192k output.mp3
# ffmpeg -i song.mp3 song.wav

file_path = sys.argv[1]
iterations = sys.argv[2]

file_name = file_path.split('.')[0]
ext = file_path.split('.')[-1]
new_name = file_name + '_transformed.' + ext
new_filename = new_name.split('.')[0]
copyfile(file_path, new_name)
rate = 320

wav2mp3_str = 'ffmpeg -y -i ' + str(new_filename) + '.wav -vn -ar 44100 -ac 2 -b:a ' \
                + str(rate) + 'k ' + str(new_filename) + '.mp3'

mp32wav_str = 'ffmpeg -y -i ' + str(new_filename) + '.mp3 ' + str(new_filename) + '.wav'

if '.wav' in new_name:
    wav2mp3 = subprocess.Popen(wav2mp3_str, shell=True)
    wav2mp3.communicate()
    wav2mp3.wait()

i = 0
for i in range(int(iterations)):
    #to wav
    mp32wav = subprocess.Popen(mp32wav_str, shell=True)
    mp32wav.communicate()
    mp32wav.wait()
    #to mp3
    wav2mp3 = subprocess.Popen(wav2mp3_str, shell=True)
    wav2mp3.communicate()
    wav2mp3.wait()

    uf.print_bar(i, int(iterations))
    i += 1
