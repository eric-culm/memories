#epochs selected for every srnn model
'''
0 = best, most clear model
1-4 = progressively degrading models
'''

#macro categories
models_map = {
                'instrumental': {},
                'fieldrec': {},
                'voice': {}
              }

#fill dict with detailed info
models_map['instrumental'] = {
                                'africanPercs' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'ambient1' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'ambient2' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'buchla' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'buchla2' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'classical' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'classical2' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'guitarAcoustic' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'guitarBaroque' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'jazz' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'organ' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'percsWar' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'percussions' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'pianoChill' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'pianoDreamy' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'pianoSmooth' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                }

models_map['fieldrec'] = {
                                'airport' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'birdsStreet' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'forest' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'industrial' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'library' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'mixed' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'office' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'rain' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'sea' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'train' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                                'wind' : {0:'x',
                                                1:'x',
                                                2:'x',
                                                3:'x',
                                                4:'x'},

                            }


models_map['voice'] = {}



print (models_map)
