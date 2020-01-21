import os
import utility_functions as uf
import subprocess
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SRNN_SR = cfg.getint('samplernn', 'samplernn_sr')
SRNN_CHUNK_SIZE = cfg.getint('samplernn', 'samplernn_chunk_size')
SRNN_DATASET_PATH = cfg.get('samplernn', 'samplernn_dataset_path')
SRNN_CODE_PATH = cfg.get('samplernn', 'samplernn_code_path')
SRNN_ENV_NAME = cfg.get('samplernn', 'samplernn_env_name')
SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')



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
               epoch_limit=100, resume=True, sample_rate=SRNN_SR, n_samples=3,
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
    command = conda_string + train_string
    print (command)
    training = subprocess.Popen(command, shell=True, cwd=code_path, stdout=subprocess.PIPE)
    training.communicate()
    training.wait()


def compute_sounds(category, model, quality=0, dur=1, num_samples=1,
                   sampling_temperature=0.95, use_cuda=True,
                   env_name=SRNN_ENV_NAME, code_path=SRNN_CODE_PATH):
    '''
    wrapper for SampleRnn sound synthesis
    '''
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

    gen_string = ' python generate_audio_user.py ' + \
                 str(params_path) + ' ' + \
                 str(model_path) + ' ' + \
                 str(dur) + ' ' + \
                 str(num_samples) + ' ' + \
                 str(sampling_temperature) + ' ' + \
                 str(output_path) + ' ' + \
                 str(use_cuda)

    command = conda_string + gen_string
    print (command)
    synthesis = subprocess.Popen(command, shell=True, cwd=code_path, stdout=subprocess.PIPE)
    synthesis.communicate()
    synthesis.wait()
    print ('sounds generated')
