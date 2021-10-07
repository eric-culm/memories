#SELECT WHICH CONFIG FILE TO LOAD FOR THE WHOLE PROJECT
import os
'''
configuration file to be loaded for the whole project
'''


configuration_file = "config/configMEMORIES_cog.ini"
#print ("Loaded config file: ", configuration_file)

def load(conf = configuration_file):
    return conf
