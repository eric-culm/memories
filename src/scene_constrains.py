import numpy as np
import sys, os
import copy
import pprint
import random
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

SRNN_DATA_PATH = cfg.get('samplernn', 'samplernn_data_path')


constrains_dict = {'sound':{},'score':{}}
parameters = {}

'''
constrains_dict['sound'] = {'category': [],
                               'model': [],
                               'variation': [],
                               'dur': []
                               }

constrains_dict['score'] = {'dur': [],
                               'volume': [],
                               'position': [],
                               'pan': [],
                               'eq': [],
                               'rev': [],
                               'rev_length': [],
                               'segment': [],
                               'stretch': [],
                               'shift': [],
                               'fade_in': [],
                               'fade_out': [],
                               }
'''

def get_constrains(constrains_list):
    '''
    join all requested constrains in one dict.
    constrains_list is a list of dicts
    '''
    output_dict = copy.deepcopy(constrains_dict)
    for curr_constrain in constrains_list:
        #curr_constrain = parameters[constrain]
        for key in curr_constrain.keys():
            for param in curr_constrain[key].keys():
                curr_value = curr_constrain[key][param]
                output_dict[key][param] = curr_value
    return output_dict

def printTree(t,s):
    a = 0
    if not isinstance(t,dict) and not isinstance(t,list):
        print ("\t"*s+str(t))
    else:
        for key in t:
            if key != 'score' and key != 'sound':
                print ("\t"*s+str(key))
                if not isinstance(t,list):
                    printTree(t[key],s+1)
                    a += 1

def gen_random_macro(verbose=False):
    p_list = []
    tot_params = len(list(parameters.keys())) - 1  #-1 subtracts the macro category
    min_params = 4
    num_params = np.arange(min_params,tot_params)
    num_params = np.random.choice(num_params)  #random number of constrains
    keys = list(parameters.keys())
    random.shuffle(keys)

    if verbose:
        print ('random constrains:')

    for i in range(num_params):
        curr_key = parameters[keys[i]]
        #print (parameters[keys[i]].keys())
        options = list(parameters[keys[i]].keys())
        sel_option = np.random.choice(options)
        p_list.append(parameters[keys[i]][sel_option])
        if verbose:
            print (str(keys[i]) + ': ' + str(sel_option))

    output_dict = get_constrains(p_list)

    return output_dict

def check_available_models(models_folder=SRNN_DATA_PATH):
    available = {}
    contents = os.listdir(models_folder)
    contents = list(filter(lambda x: '.DS_Store' not in x, contents))
    #check is a model folder contains the precomputed sounds
    for cat in contents:
        available[cat] = []
        category_path = os.path.join(models_folder, cat)
        models = os.listdir(category_path)
        models = list(filter(lambda x: '.DS_Store' not in x, models))
        for mod in models:
            mod_path = os.path.join(category_path, mod)
            ins = os.listdir(mod_path)
            if 'sounds' in ins:
                available[cat].append(mod)
    #cut empty categories
    empties = []
    for cat in available:
        if available[cat] == []:
            empties.append(cat)
    for e in empties:
        del available[e]

    return available





#SOUND SELECTION CONSTRAINS
parameters['selection_type'] = {}
parameters['selection_length'] = {}


#prefer highest quality
parameters['selection_type']['more_hq'] = copy.deepcopy(constrains_dict)
def phq(in_list):
    in_list = list(in_list)
    for i in range(len(in_list)):
        in_list.append(0)
    return in_list
parameters['selection_type']['more_hq']['sound']['variation'] = phq

#prefer lowest quality
parameters['selection_type']['more_lq'] = copy.deepcopy(constrains_dict)
def plq(in_list):
    in_list = list(in_list)
    for i in range(len(in_list)):
        list.append(max(in_list))
    return in_list
parameters['selection_type']['more_lq']['sound']['variation'] = phq

#prefer mid quality
parameters['selection_type']['more_mq'] = copy.deepcopy(constrains_dict)
def phmq(in_list):
    in_list = list(in_list)
    to_append = in_list.copy()
    to_append.remove(max(to_append))
    to_append.remove(min(to_append))
    for i in range(len(list)):
        in_list.append(to_append)
    return in_list
parameters['selection_type']['more_mq']['sound']['variation'] = phq

#only long sounds selected
parameters['selection_length'] = {}
parameters['selection_length']['long'] = copy.deepcopy(constrains_dict)
parameters['selection_length']['long']['sound']['dur'] = lambda x: [60,60,30]

#only short sounds selected
parameters['selection_length']['short'] = copy.deepcopy(constrains_dict)
parameters['selection_length']['short']['sound']['dur'] = lambda x: [10,5,3]

#SCORE CONSTRAINS
parameters['length'] = {}
#only sounds longer than 40 sec
parameters['length']['very_long'] = copy.deepcopy(constrains_dict)
parameters['length']['very_long']['score']['dur'] = lambda x: np.arange(40,60,0.1)

#only sounds between 10 and 40
parameters['length']['long'] = copy.deepcopy(constrains_dict)
parameters['length']['long']['score']['dur'] = lambda x: np.arange(10,40,0.01)

#only sounds between 5,20
parameters['length']['short'] = copy.deepcopy(constrains_dict)
parameters['length']['short']['score']['dur'] = lambda x: np.arange(1,10,0.01)

#only sounds between 0.2 and 1
parameters['length']['very_short'] = copy.deepcopy(constrains_dict)
parameters['length']['very_short']['score']['dur'] = lambda x: np.arange(0.2,1,0.01)

#VOLUME
parameters['volume'] = {}
#hi volume
parameters['volume']['high'] = copy.deepcopy(constrains_dict)
parameters['volume']['high']['score']['volume'] = lambda x: np.arange(0.7,1,0.01)

#mid volume
parameters['volume']['mid'] = copy.deepcopy(constrains_dict)
parameters['volume']['mid']['score']['volume'] = lambda x: np.arange(0.3,0.7,0.01)

#low volume
parameters['volume']['low'] = copy.deepcopy(constrains_dict)
parameters['volume']['low']['score']['volume'] = lambda x: np.arange(0.1,0.3,0.01)

#POSITION
parameters['position'] = {}
#position beginning
parameters['position']['at_beginning'] = copy.deepcopy(constrains_dict)
parameters['position']['at_beginning']['score']['position'] = lambda x: [0]

#position beginning
parameters['position']['initial'] = copy.deepcopy(constrains_dict)
parameters['position']['initial']['score']['position'] = lambda x: np.arange(0,0.3,0.01)

#position mid
parameters['position']['mid'] = copy.deepcopy(constrains_dict)
parameters['position']['mid']['score']['position'] = lambda x: np.arange(0.3,0.7,0.01)

#position ending
parameters['position']['ending'] = copy.deepcopy(constrains_dict)
parameters['position']['ending']['score']['position'] = lambda x: np.arange(0.7,1.,0.01)

#PAN
parameters['pan'] = {}
#around center
parameters['pan']['around_center'] = copy.deepcopy(constrains_dict)
parameters['pan']['around_center']['score']['pan'] = lambda x: np.arange(-0.2,0.2,0.01)

#towards left
parameters['pan']['towards_left'] = copy.deepcopy(constrains_dict)
parameters['pan']['towards_left']['score']['pan'] = lambda x: np.arange(-1,0.2,0.01)

#towards right
parameters['pan']['towards_right'] = copy.deepcopy(constrains_dict)
parameters['pan']['towards_right']['score']['pan'] = lambda x: np.arange(0.2,1,0.01)

#always left
parameters['pan']['always_left'] = copy.deepcopy(constrains_dict)
parameters['pan']['always_left']['score']['pan'] = lambda x: [-1]

#always right
parameters['pan']['always_right'] = copy.deepcopy(constrains_dict)
parameters['pan']['always_right']['score']['pan'] = lambda x: [1]

#EQ
parameters['eq'] = {}
#eq always on
parameters['eq']['always'] = copy.deepcopy(constrains_dict)
parameters['eq']['always']['score']['eq'] = lambda x: [True]

#more probable eq
parameters['eq']['more'] = copy.deepcopy(constrains_dict)
parameters['eq']['more']['score']['eq'] = lambda x: [True, True, False]

#eq always on
parameters['eq']['never'] = copy.deepcopy(constrains_dict)
parameters['eq']['never']['score']['eq'] = lambda x: [False]

#more probable not eq
parameters['eq']['less'] = copy.deepcopy(constrains_dict)
parameters['eq']['less']['score']['eq'] = lambda x: [False, True, False]

#REV
parameters['rev_prob'] = {}
parameters['rev_length'] = {}
#rev always on
parameters['rev_prob'] = {}
parameters['rev_prob']['always'] = copy.deepcopy(constrains_dict)
parameters['rev_prob']['always']['score']['rev'] = lambda x: [True]

#more probable rev
parameters['rev_prob']['more'] = copy.deepcopy(constrains_dict)
parameters['rev_prob']['more']['score']['rev'] = lambda x: [True, True, False]

#rev always on
parameters['rev_prob']['never'] = copy.deepcopy(constrains_dict)
parameters['rev_prob']['never']['score']['rev'] = lambda x: [False]

#more probable not rev
parameters['rev_prob']['less'] = copy.deepcopy(constrains_dict)
parameters['rev_prob']['less']['score']['rev_length'] = lambda x: [False, True, False]

#always long rev
parameters['rev_length'] = {}
parameters['rev_length']['long'] = copy.deepcopy(constrains_dict)
parameters['rev_length']['long']['score']['rev_length'] = lambda x: sorted(x)[-2:]

#always short rev
parameters['rev_length']['short'] = copy.deepcopy(constrains_dict)
parameters['rev_length']['short']['score']['rev_length'] = lambda x: sorted(x)[:2]

#always mid reverb length
def rmm(in_list):
    in_list = list(in_list)
    in_list.remove(min(in_list))
    in_list.remove(max(in_list))
    return in_list
parameters['rev_length']['mid'] = copy.deepcopy(constrains_dict)
parameters['rev_length']['mid']['score']['rev_length'] = rmm

#SEGMENT
parameters['segment'] = {}
#segment always on
parameters['segment']['always'] = copy.deepcopy(constrains_dict)
parameters['segment']['always']['score']['segment'] = lambda x: [True]

#more probable segment
parameters['segment']['more'] = copy.deepcopy(constrains_dict)
parameters['segment']['more']['score']['segment'] = lambda x: [True, True, False]

#segment always on
parameters['segment']['never'] = copy.deepcopy(constrains_dict)
parameters['segment']['never']['score']['segment'] = lambda x: [False]

#more probable not segment
parameters['segment']['less'] = copy.deepcopy(constrains_dict)
parameters['segment']['less']['score']['segment'] = lambda x: [False, True, False]

#STRETCH
parameters['stretch'] = {}
#stretch long
parameters['stretch']['long'] = copy.deepcopy(constrains_dict)
parameters['stretch']['long']['score']['stretch'] = lambda x: [y for y in x if y>4]

#stretch short
parameters['stretch']['short'] = copy.deepcopy(constrains_dict)
parameters['stretch']['short']['score']['stretch'] = lambda x: [y for y in x if y<0.3]

#stretch mid
parameters['stretch']['mid'] = copy.deepcopy(constrains_dict)
parameters['stretch']['mid']['score']['stretch'] = lambda x: [y for y in x if y>0.5 and y<2]

#no stretch
parameters['stretch']['never'] = copy.deepcopy(constrains_dict)
parameters['stretch']['never']['score']['stretch'] = lambda x: [1]

#SHIFT
parameters['shift'] = {}
#shift lo
parameters['shift']['low'] = copy.deepcopy(constrains_dict)
parameters['shift']['low']['score']['shift'] = lambda x: [y for y in x if y<30]

#shift hi
parameters['shift']['high'] = copy.deepcopy(constrains_dict)
parameters['shift']['high']['score']['shift'] = lambda x: [y for y in x if y>12]

#shift mid
parameters['shift']['mid'] = copy.deepcopy(constrains_dict)
parameters['shift']['mid']['score']['shift'] = lambda x: [y for y in x if y>-10 and y<10]

#no shift
parameters['shift']['never'] = copy.deepcopy(constrains_dict)
parameters['shift']['never']['score']['shift'] = lambda x: [1]

#FADES
parameters['fade_in'] = {}
parameters['fade_out'] = {}
#short fade in
parameters['fade_in']['short'] = copy.deepcopy(constrains_dict)
parameters['fade_in']['short']['score']['fade_in'] = lambda x: np.arange(0,50,0.1)

#short fade out
parameters['fade_out']['short'] = copy.deepcopy(constrains_dict)
parameters['fade_out']['short']['score']['fade_out'] = lambda x: np.arange(0,50,0.1)

#long fade in
parameters['fade_in']['long'] = copy.deepcopy(constrains_dict)
parameters['fade_in']['long']['score']['fade_in'] = lambda x: np.arange(max(x)*0.7, max(x), 0.1)

#long fade out
parameters['fade_out']['long'] = copy.deepcopy(constrains_dict)
parameters['fade_out']['long']['score']['fade_out'] = lambda x: np.arange(max(x)*0.7, max(x), 0.1)

#mid fade in
parameters['fade_in']['mid'] = copy.deepcopy(constrains_dict)
parameters['fade_in']['mid']['score']['fade_in'] = lambda x: np.arange(max(x)*0.3, max(x)*0.7, 0.1)

#mid fade out
parameters['fade_out']['mid'] = copy.deepcopy(constrains_dict)
parameters['fade_out']['mid']['score']['fade_out'] = lambda x: np.arange(max(x)*0.3, max(x)*0.7, 0.1)



#MACROS
#combinations of constrains
parameters['macro'] = {}
p = parameters.copy()

#sound particle
parameters['macro']['particle'] = copy.deepcopy(constrains_dict)
particle_features = [p['selection_length']['short'], p['length']['very_short'],
                    p['volume']['high'], p['eq']['less'], p['stretch']['never'],
                    p['fade_in']['short'], p['fade_out']['mid']]
parameters['macro']['particle'] = get_constrains(particle_features)


#sound particle
parameters['macro']['long_low'] = copy.deepcopy(constrains_dict)
long_low_features = [p['selection_length']['long'], p['length']['very_long'],
                    p['position']['at_beginning'],
                    p['volume']['mid'], p['eq']['less'],p['shift']['low'],
                    p['fade_in']['long'], p['fade_out']['long']]
parameters['macro']['long_low'] = get_constrains(long_low_features)
