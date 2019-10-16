import os
import subprocess
import configparser
import loadconfig

#load config variables
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

path = cfg.get('osc', 'client_path')
path = os.path.join(path, 'client')
client_ip = cfg.get('osc', 'client_ip')

string = 'scp -r eric@' + client_ip + ':' + path + ' ../max_code'

proc = subprocess.Popen(string, shell=True)
proc.communicate()
proc.wait()

print ('\nall files copied')
