import os
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
SRNN_ENV_PATH = cfg.get('samplernn', 'samplernn_env_path')

def split_audio(input_file, chunk_size=SRNN_CHUNK_SIZE, sr=SRNN_SR):
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

def train_srnn(code_path=SRNN_CODE_PATH, env_path=SRNN_ENV_PATH):
    conda_string = 'conda run -p ' + env_path
