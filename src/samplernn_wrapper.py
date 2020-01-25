import os, sys
import utility_functions as uf
import subprocess
import configparser
import loadconfig
from srnn_models_map import *
import shutil
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile
from timeit import timeit
import time
import resource



config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SRNN_SR = cfg.getint('samplernn', 'samplernn_sr')
SRNN_CHUNK_SIZE = cfg.getint('samplernn', 'samplernn_chunk_size')
SRNN_DATASET_PATH = cfg.get('samplernn', 'samplernn_dataset_path')
SRNN_CODE_PATH = cfg.get('samplernn', 'samplernn_code_path')
SRNN_ENV_NAME = cfg.get('samplernn', 'samplernn_env_name')
SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')
MAIN_SR = cfg.getint('main', 'sr_processing')



srnn_categories = {}

def split_audio(input_file, chunk_size=SRNN_CHUNK_SIZE, sr=SRNN_SR):
    '''
    load audio file, convert it to wav with given sr and splice it
    in equal chunks
    '''
    dataset_name = os.path.basename(input_file).split('.')[0]
    dataset_name = dataset_name + '_' + str(chunk_size) + '_' + str(sr)
    split_script = SRNN_DATASET_PATH + '/split_file.sh'
    args =[' bash split_file.sh', str(input_file), str(chunk_size), str(sr), dataset_name]
    cd = 'cd ' + SRNN_DATASET_PATH + ';'
    command = ' '.join(args)
    command = cd + command
    print (command)
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    process.wait()


def split_all_files(input_folder, chunk_size=SRNN_CHUNK_SIZE, sr=SRNN_SR):
    '''
    split all audio files in a folder
    '''
    contents = os.listdir(input_folder)
    contents = [os.path.join(input_folder, x) for x in contents]
    i = 0
    for sound in contents:
        split_audio(sound)
        uf.print_bar(i, len(contents))
        i += 1

def train_srnn(input_dataset, frame_sizes='16 4', n_rnn=2, batch_size=128, keep_old_checkpoints=True,
               gpu_id=0, epoch_limit=100, resume=True, sample_rate=SRNN_SR, n_samples=3,
               sample_length=2, sampling_temperature=0.95, env_name=SRNN_ENV_NAME,
               code_path=SRNN_CODE_PATH):
    '''
    wrapper for SampleRNN training
    '''
    if input_dataset[-1] == '/':
        input_dataset = input_dataset[:-1]
    #exp_name = os.path.basename(input_dataset).split('.')[0]
    exp_name = input_dataset
    sample_length = sample_length * sample_rate
    conda_string = 'conda run -n ' + str(env_name)
    cuda_string = ' CUDA_VISIBLE_DEVICES=' + str(gpu_id)
    train_string = ' python train.py' + \
                   ' --exp ' + '"' + str(exp_name) + '"' + \
                   ' --frame_sizes ' + str(frame_sizes) + \
                   ' --n_rnn ' + str(n_rnn) + \
                   ' --batch_size ' + str(batch_size) + \
                   ' --keep_old_checkpoints ' + str(keep_old_checkpoints) + \
                   ' --resume ' + str(resume) + \
                   ' --sample_rate ' + str(sample_rate) + \
                   ' --n_samples ' + str(n_samples) + \
                   ' --sample_length ' + str(sample_length) + \
                   ' --sampling_temperature ' + str(sampling_temperature) + \
                   ' --epoch_limit ' + str(epoch_limit) + \
                   ' --dataset ' + str(input_dataset)

    command = conda_string + cuda_string + train_string

    print (command)

    training = subprocess.Popen(command, shell=True, cwd=code_path, stdout=subprocess.PIPE)
    training.communicate()
    training.wait()



def prepare_sound(input_sound):
    '''
    -remove DC
    -apply in and out fades
    -normalize
    -convert to mono wav pcm 44100 16bit
    '''
    #remove dc
    samples, dummy = librosa.core.load(input_sound, sr=MAIN_SR)
    samples = signal.detrend(samples)
    sos = signal.butter(10, 15, 'hp', fs=MAIN_SR, output='sos')
    samples = signal.sosfilt(sos, samples)

    #apply fades
    in_ade_length = 10
    out_fade_length = in_ade_length * 10
    mask = np.ones(len(samples))
    ramp1 = np.arange(in_ade_length) / in_ade_length
    ramp2 = np.arange(out_fade_length) / (out_fade_length)
    ramp2 = np.array(np.flip(ramp2))
    mask[:in_ade_length] = ramp1
    mask[-out_fade_length:] = ramp2
    faded = samples * mask

    #normalize
    samples = samples / np.max(samples)
    samples = samples * 0.8

    #write file
    soundfile.write(input_sound, samples, 44100, format='WAV', subtype='PCM_16')

def cazzo(output_path):
    print ('converting...')
    contents = os.listdir(output_path)
    contents = list(filter(lambda x: '.wav' in x, contents))
    tot = len(contents)
    index = 0
    for i in contents:
        curr_sound = os.path.abspath(os.path.join(output_path, i))
        prepare_sound(curr_sound)
        index +=1
        uf.print_bar(index, tot)


def generate_sounds(category, model, quality=0, dur=1, num_samples=1,
                   sampling_temperature=0.95, use_cuda=True, gpu_id=0,
                   env_name=SRNN_ENV_NAME, code_path=SRNN_CODE_PATH):
    '''
    wrapper for SampleRnn sound synthesis
    '''
    time_start = time.clock()
    model_name = str(model) + '_' + str(quality)

    base_path = os.path.join(SRNN_DATA_PATH, category, model)
    model_path = os.path.join(base_path, 'models', model_name)
    model_path = os.path.abspath(model_path)

    params_path = os.path.join(base_path, 'models', 'sample_rnn_params.json')
    params_path = os.path.abspath(params_path)

    output_path = os.path.join(base_path, 'sounds', 'dur_' + str(dur))
    output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #arguments order:
    #PARAMS_PATH, PRETRAINED_PATH, DUR, NUM_SOUNDS, SAMPLING_TEMPERATURE, GENERATED_PATH, USE_CUDA

    conda_string = 'conda run -n ' + str(env_name)
    cuda_string = ' CUDA_VISIBLE_DEVICES=' + str(gpu_id)
    gen_string = ' python generate_audio_user.py ' + \
                 str(params_path) + ' ' + \
                 str(model_path) + ' ' + \
                 str(dur) + ' ' + \
                 str(num_samples) + ' ' + \
                 str(sampling_temperature) + ' ' + \
                 str(output_path) + ' ' + \
                 str(use_cuda)

    if use_cuda:
        command = conda_string + cuda_string + gen_string
    else:
        command = conda_string + gen_string


    print (command)
    print ('')
    print ('generating sounds with sampleRNN...')

    synthesis = subprocess.Popen(command, shell=True, cwd=code_path, stdout=subprocess.PIPE)
    synthesis.communicate()
    synthesis.wait()

    #prepare sound
    time.sleep(1)
    print ('converting...')
    contents = os.listdir(output_path)
    contents = list(filter(lambda x: '.wav' in x, contents))
    tot = len(contents)
    index = 0
    for i in contents:
        curr_sound = os.path.abspath(os.path.join(output_path, i))
        prepare_sound(curr_sound)
        index +=1
        uf.print_bar(index, tot)
    #print ('sounds generated')
    time_elapsed = (time.clock() - time_start)
    string = 'Generated ' + str(num_samples) + ' of ' + str(dur) + ' seconds in ' + str(time_elapsed) + ' seconds'
    print ('')
    print (string)
    info = resource.getrusage(resource.RUSAGE_CHILDREN)
    print (info)
def move_selected_models(input_folder, category, model):
    '''
    copy selected model epochs to srnn_data_path
    --selected models are defined in srnn_models_map script
    '''
    #model_name = str(model) + '_' + str(quality)

    base_path = os.path.join(SRNN_DATA_PATH, category, model, 'models')
    params_out = os.path.join(base_path, 'sample_rnn_params.json')
    params_out = os.path.abspath(params_out)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    #correct folder name
    contents = os.listdir(input_folder)
    for folder in contents:
        if folder != '.DS_Store':
            name = folder.split(':')[1].split('-')[0]
            curr_category = name.split('_')[0]
            curr_model = name.split('_')[1]
            if category == curr_category and model == curr_model:
                break
    folder = os.path.join(input_folder, folder)
    models_in = os.path.join(input_folder, folder, 'checkpoints')
    models_in = os.path.abspath(models_in)

    #dict of model names
    model_names = {}
    contents = os.listdir(models_in)
    for i in range(len(contents)):
        epoch = contents[i].split('-')[-2][2:]
        model_names[epoch] = contents[i]


    #move parameters json
    params_in = os.path.join(folder, 'sample_rnn_params.json')
    shutil.copy(params_in, params_out)

    #extract and copy correct epochs
    selected_epochs = models_map[category][model]
    for i in selected_epochs.keys():
        curr_in_model_name = model_names[str(selected_epochs[i])]
        curr_in_model_name = os.path.join(models_in, curr_in_model_name)
        curr_in_model_name = os.path.abspath(curr_in_model_name)
        curr_out_model_name = str(model) + '_' + str(i)
        curr_out_model_name = os.path.join(base_path, curr_out_model_name)
        curr_out_model_name = os.path.abspath(curr_out_model_name)
        string = 'transferring: ' + str(curr_in_model_name.split('/')[-1]) + ' as ' + str(curr_out_model_name.split('/')[-1])
        print (string)
        shutil.copy(curr_in_model_name, curr_out_model_name)
    print ('transfer completed')
