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

def train_srnn(input_dataset, frame_sizes='16 4', n_rnn=2, batch_size=128, keep_old_checkpoints=False,
               epoch_limit=1000, resume=True, sample_rate=SRNN_SR, n_samples=1,
               sample_length=2, sampling_temperature=0.95, env_name=SRNN_ENV_NAME,
               code_path=SRNN_CODE_PATH):
    '''
    wrapper for SampleRNN training
    '''
    exp_name = input_dataset.split('/')[-1]
    sample_length = sample_length * sample_rate
    conda_string = 'conda run -n ' + str(env_name)
    train_string = ' python ' + os.path.join(code_path, 'train.py') + \
                   ' --exp ' + '"' + str(exp_name) + '"' + \
                   ' --frame_sizes ' + str(frame_sizes) + \
                   ' --n_rnn ' + str(n_rnn) + \
                   ' --batch_size ' + str(batch_size) + \
                   ' --keep_old_checkpoints ' + str(keep_old_checkpoints) + \
                   ' --epoch_limit ' + str(epoch_limit) + \
                   ' --resume ' + str(resume) + \
                   ' --sample_rate ' + str(sample_rate) + \
                   ' --n_samples ' + str(n_samples) + \
                   ' --sample_length ' + str(sample_length) + \
                   ' --sampling_temperature ' + str(sampling_temperature) + \
                   ' --dataset ' + str(input_dataset)
    command = conda_string + train_string
    print (command)
    training = subprocess.Popen(command, shell=True)
    training.communicate()
    training.wait()
