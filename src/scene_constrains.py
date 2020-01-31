import numpy as np
import sys
import copy

constrains_dict = {'sound':{},'score':{}}
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
    join all requested constrains in one dict
    '''
    output_dict = copy.deepcopy(constrains_dict)
    for constrain in constrains_list:
        curr_constrain = eval(constrain)
        for key in curr_constrain.keys():
            for param in curr_constrain[key].keys():
                curr_value = curr_constrain[key][param]
                output_dict[key][param] = curr_value
    return output_dict

#only available
#only_available = constrains_dict.copy()
only_available = copy.deepcopy(constrains_dict)
only_available['sound']['category'] = lambda x: ['instrumental']
only_available['sound']['model'] = lambda x: ['africanPercs', 'ambient1', 'buchla',
                                            'buchla2', 'classical', 'classical2',
                                            'guitarAcoustic', 'jazz', 'organ', 'percsWar',
                                            'percussions', 'pianoChill']


#SOUND SELECTION CONSTRAINS
#prefer highest quality
prefer_hq = copy.deepcopy(constrains_dict)
def phq(in_list):
    for i in range(len(in_list)):
        in_list.append(0)
    return in_list
prefer_hq['sound']['variation'] = phq

#prefer lowest quality
prefer_lq = copy.deepcopy(constrains_dict)
def plq(in_list):
    for i in range(len(in_list)):
        list.append(max(in_list))
    return in_list
prefer_lq['sound']['variation'] = phq

#prefer mid quality
prefer_mq = copy.deepcopy(constrains_dict)
def phmq(in_list):
    to_append = in_list.copy()
    to_append.remove(max(to_append))
    to_append.remove(min(to_append))
    for i in range(len(list)):
        in_list.append(to_append)
    return in_list
prefer_mq['sound']['variation'] = phq

#only long sounds selected
only_long_selected = copy.deepcopy(constrains_dict)
only_long_selected['sound']['dur'] = lambda x: [60,60,30]

#only short sounds selected
only_short_selected = copy.deepcopy(constrains_dict)
only_short_selected['sound']['dur'] = lambda x: [10,5,3]

#SCORE CONSTRAINS

#only sounds longer than 40 sec
very_long_scored = copy.deepcopy(constrains_dict)
very_long_scored['score']['dur'] = lambda x: np.arange(40,60,0.1)

#only sounds between 10 and 40
long_scored = copy.deepcopy(constrains_dict)
long_scored['score']['dur'] = lambda x: np.arange(10,40,0.01)

#only sounds between 5,20
short_scored = copy.deepcopy(constrains_dict)
short_scored['score']['dur'] = lambda x: np.arange(1,10,0.01)

#only sounds between 0.2 and 1
very_short_scored = copy.deepcopy(constrains_dict)
very_short_scored['score']['dur'] = lambda x: np.arange(0.2,1,0.01)

#hi volume
volume_hi = copy.deepcopy(constrains_dict)
volume_hi['score']['volume'] = lambda x: np.arange(0.7,1,0.01)

#mid volume
volume_mid = copy.deepcopy(constrains_dict)
volume_mid['score']['volume'] = lambda x: np.arange(0.3,0.7,0.01)

#low volume
volume_low = copy.deepcopy(constrains_dict)
volume_low['score']['volume'] = lambda x: np.arange(0.1,0.3,0.01)

#around center
pan_around_center = copy.deepcopy(constrains_dict)
pan_around_center['score']['pan'] = lambda x: np.arange(-0.2,0.2,0.01)

#towards left
pan_towards_left = copy.deepcopy(constrains_dict)
pan_towards_left['score']['pan'] = lambda x: np.arange(-0.2,-1,0.01)

#towards right
pan_towards_right = copy.deepcopy(constrains_dict)
pan_towards_right['score']['pan'] = lambda x: np.arange(0.2,1,0.01)

#always left
pan_always_left = copy.deepcopy(constrains_dict)
pan_always_left['score']['pan'] = lambda x: [-1]

#always right
pan_always_right = copy.deepcopy(constrains_dict)
pan_always_right['score']['pan'] = lambda x: [1]

#eq always on
always_eq = copy.deepcopy(constrains_dict)
always_eq['score']['eq'] = lambda x: [True]

#more probable eq
more_eq = copy.deepcopy(constrains_dict)
more_eq['score']['eq'] = lambda x: [True, True, False]

#eq always on
never_eq = copy.deepcopy(constrains_dict)
never_eq['score']['eq'] = lambda x: [False]

#more probable not eq
less_eq = copy.deepcopy(constrains_dict)
less_eq['score']['eq'] = lambda x: [False, True, False]

#rev always on
always_rev = copy.deepcopy(constrains_dict)
always_rev['score']['rev'] = lambda x: [True]

#more probable rev
more_rev = copy.deepcopy(constrains_dict)
more_rev['score']['rev'] = lambda x: [True, True, False]

#rev always on
never_rev = copy.deepcopy(constrains_dict)
never_rev['score']['rev'] = lambda x: [False]

#more probable not rev
less_rev = copy.deepcopy(constrains_dict)
less_rev['score']['rev'] = lambda x: [False, True, False]

#always long rev
always_long_rev = copy.deepcopy(constrains_dict)
always_long_rev['score']['rev'] = lambda x: sorted(x)[-2:]

#always short rev
always_short_rev = copy.deepcopy(constrains_dict)
always_short_rev['score']['rev'] = lambda x: sorted(x)[:2]

#always mid reverb length
def rmm(in_list):
    in_list.remove(min(in_list))
    in_list.remove(max(in_list))
    return in_list
always_mid_rev = copy.deepcopy(constrains_dict)
always_mid_rev['score']['rev'] = rmm

#segment always on
always_segment = copy.deepcopy(constrains_dict)
always_segment['score']['segment'] = lambda x: [True]

#more probable segment
more_segment = copy.deepcopy(constrains_dict)
more_segment['score']['segment'] = lambda x: [True, True, False]

#segment always on
never_segment = copy.deepcopy(constrains_dict)
never_segment['score']['segment'] = lambda x: [False]

#more probable not segment
less_segment = copy.deepcopy(constrains_dict)
less_segment['score']['segment'] = lambda x: [False, True, False]

#stretch long
stretch_long = copy.deepcopy(constrains_dict)
stretch_long['score']['stretch'] = lambda x: [y for y in x if y>4]

#stretch short
stretch_short = copy.deepcopy(constrains_dict)
stretch_short['score']['stretch'] = lambda x: [y for y in x if y<0.3]

#stretch long
stretch_long = copy.deepcopy(constrains_dict)
stretch_long['score']['stretch'] = lambda x: [y for y in x if y>4]

#stretch mid
stretch_mid = copy.deepcopy(constrains_dict)
stretch_mid['score']['stretch'] = lambda x: [y for y in x if y>0.5 and y<2]

#no stretch
no_stretch = copy.deepcopy(constrains_dict)
no_stretch['score']['stretch'] = lambda x: [1]

#shift lo
shift_low = copy.deepcopy(constrains_dict)
shift_low['score']['shift'] = lambda x: [y for y in x if y<30]

#shift hi
shift_hi = copy.deepcopy(constrains_dict)
shift_hi['score']['shift'] = lambda x: [y for y in x if y>12]

#shift mid
shift_mid = copy.deepcopy(constrains_dict)
shift_mid['score']['shift'] = lambda x: [y for y in x if y>-10 and y<10]

#no shift
no_shift = copy.deepcopy(constrains_dict)
no_shift['score']['shift'] = lambda x: [1]

#short fade in
fade_in_short = copy.deepcopy(constrains_dict)
fade_in_short['score']['fade_in'] = lambda x: [y for y in x if y < 50]

#short fade out
fade_out_short = copy.deepcopy(constrains_dict)
fade_out_short['score']['fade_out'] = lambda x: [y for y in x if y < 50]

#long fade in
fade_in_long = copy.deepcopy(constrains_dict)
fade_in_long['score']['fade_in'] = lambda x: [y for y in x if y > max(x)*0.7]

#long fade out
fade_out_long = copy.deepcopy(constrains_dict)
fade_out_long['score']['fade_out'] = lambda x: [y for y in x if y > max(x)*0.7]

#mid fade in
fade_in_mid = copy.deepcopy(constrains_dict)
fade_in_mid['score']['fade_in'] = lambda x: [y for y in x if y<max(x)*0.7 and y>max(x)*0.3]

#mid fade out
fade_out_mid = copy.deepcopy(constrains_dict)
fade_out_mid['score']['fade_out'] = lambda x: [y for y in x if y<max(x)*0.7 and y>max(x)*0.3]
#MACROS
#combinations of constrains
#sound particel
particle = copy.deepcopy(constrains_dict)
particle_features = ['only_short_selected', 'very_short_scored', 'volume_hi',
                     'less_eq', 'no_stretch', 'fade_in_short',
                     'fade_out_mid']
particel = get_constrains(particle_features)
