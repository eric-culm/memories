import numpy as np

constrains_dict = {'sound':{},
                      'score':{}
                      }
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

#only available
only_available = constrains_dict.copy()
only_available['sound']['category'] = lambda x: ['instrumental']
only_available['sound']['model'] = lambda x: ['africanPercs', 'ambient1', 'buchla',
                                            'buchla2', 'classical', 'classical2',
                                            'guitarAcoustic', 'jazz', 'organ', 'percsWar',
                                            'percussions', 'pianoChill']


#SOUND SELECTION CONSTRAINS
#prefer highest quality
prefer_hq = constrains_dict.copy()
def phq(in_list):
    for i in range(len(in_list)):
        in_list.append(0)
    return in_list
prefer_hq['sound']['variation'] = phq

#prefer lowest quality
prefer_lq = constrains_dict.copy()
def plq(in_list):
    for i in range(len(in_list)):
        list.append(max(in_list))
    return in_list
prefer_lq['sound']['variation'] = phq

#prefer mid quality
prefer_mq = constrains_dict.copy()
def phmq(in_list):
    to_append = in_list.copy()
    to_append.remove(max(to_append))
    to_append.remove(min(to_append))
    for i in range(len(list)):
        in_list.append(to_append
    return in_list
prefer_mq['sound']['variation'] = phq

#only long sounds selected
only_long_selected = constrains_dict.copy()
only_long_selected['sound']['dur'] = lambda x: [60,60,30]

#only short sounds selected
only_short_selected = constrains_dict.copy()
only_short_selected['sound']['dur'] = lambda x: [10,5,3]

#SCORE CONSTRAINS

#only sounds longer than 40 sec
very_long_scored = constrains_dict.copy()
very_long_scored['score']['dur'] = lambda x: np.arange(40,60,0.1)

#only sounds between 10 and 40
long_scored = constrains_dict.copy()
long_scored['score']['dur'] = lambda x: np.arange(10,40,0.01)

#only sounds between 5,20
short_scored = constrains_dict.copy()
short_scored['score']['dur'] = lambda x: np.arange(1,10,0.01)

#only sounds between 0.2 and 1
very_short_scored = constrains_dict.copy()
very_short_scored['score']['dur'] = lambda x: np.arange(0.2,1,0.01)

#hi volume
hi_volume = constrains_dict.copy()
hi_volume['score']['volume'] = lambda x: np.arange(0.7,1,0.01)

#mid volume
mid_volume = constrains_dict.copy()
mid_volume['score']['volume'] = lambda x: np.arange(0.3,0.7,0.01)

#low volume
mid_volume = constrains_dict.copy()
mid_volume['score']['volume'] = lambda x: np.arange(0.1,0.3,0.01)

#around center
around_center = constrains_dict.copy()
around_center['score']['pan'] = lambda x: np.arange(-0.2,0.2,0.01)

#towards left
towards_left = constrains_dict.copy()
towards_left['score']['pan'] = lambda x: np.arange(-0.2,-1,0.01)

#towards right
towards_right = constrains_dict.copy()
towards_right['score']['pan'] = lambda x: np.arange(0.2,1,0.01)

#always left
always_left = constrains_dict.copy()
always_left['score']['pan'] = lambda x: [-1]

#always right
always_right = constrains_dict.copy()
always_right['score']['pan'] = lambda x: [1]

#eq always on
always_eq = constrains_dict.copy()
always_eq['score']['eq'] = lambda x: [True]

#more probable eq
more_eq = constrains_dict.copy()
more_eq['score']['eq'] = lambda x: [True, True, False]

#eq always on
never_eq = constrains_dict.copy()
never_eq['score']['eq'] = lambda x: [False]

#more probable not eq
less_eq = constrains_dict.copy()
less_eq['score']['eq'] = lambda x: [False, True, False]

#rev always on
always_rev = constrains_dict.copy()
always_rev['score']['rev'] = lambda x: [True]

#more probable rev
more_rev = constrains_dict.copy()
more_rev['score']['rev'] = lambda x: [True, True, False]

#rev always on
never_rev = constrains_dict.copy()
never_rev['score']['rev'] = lambda x: [False]

#more probable not rev
less_rev = constrains_dict.copy()
less_rev['score']['rev'] = lambda x: [False, True, False]

#always long rev
always_long_rev = constrains_dict.copy()
always_long_rev['score']['rev'] = lambda x: sorted(x)[-2:]

#always short rev
always_short_rev = constrains_dict.copy()
always_short_rev['score']['rev'] = lambda x: sorted(x)[:2]

#always mid reverb length
def rmm(in_list):
    in_list.remove(min(in_list))
    in_list.remove(max(in_list))
    return in_list
always_mid_rev = constrains_dict.copy()
always_short_rev['score']['rev'] = rmm

#segment always on
always_segment = constrains_dict.copy()
always_segment['score']['segment'] = lambda x: [True]

#more probable segment
more_segment = constrains_dict.copy()
more_segment['score']['segment'] = lambda x: [True, True, False]

#segment always on
never_segment = constrains_dict.copy()
never_segment['score']['segment'] = lambda x: [False]

#more probable not segment
less_segment = constrains_dict.copy()
less_segment['score']['segment'] = lambda x: [False, True, False]

#stretch long
stretch_long = constrains_dict.copy()
stretch_long['score']['stretch'] = lambda x: x > 4.

#stretch short
stretch_short = constrains_dict.copy()
stretch_short['score']['stretch'] = lambda x: x < 0.3

#stretch long
stretch_long = constrains_dict.copy()
stretch_long['score']['stretch'] = lambda x: x > 4.

#stretch mid
stretch_mid = constrains_dict.copy()
stretch_mid['score']['stretch'] = lambda x: x > 0.5 and x < 2

#no stretch
no_stretch = constrains_dict.copy()
no_stretch['score']['stretch'] = lambda x: [1]

#shift lo
shift_low = constrains_dict.copy()
shift_low['score']['shift'] = lambda x: x < -30

#shift hi
shift_hi = constrains_dict.copy()
shift_hi['score']['shift'] = lambda x: x > 12

#shift mid
shift_mid = constrains_dict.copy()
shift_mid['score']['shift'] = lambda x: x > -10 and x < 10












def get_constrains(constrains_list):
    '''
    join all requested constrains in one dict
    '''
    d = constrains_dict
    for constrain in constrains_list:
        d.update(eval(constrain))
    return d
