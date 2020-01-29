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

#prefer highest quality
prefer_hq = constrains_dict.copy()
def phq(list):
    for i in range(len(list)):
        list.append(0)
    return list
prefer_hq['sound']['variation'] = phq



def get_constrains(constrains_list):
    '''
    join all requested constrains in one dict
    '''
    d = constrains_dict
    for constrain in constrains_list:
        d.update(eval(constrain))
    return d
