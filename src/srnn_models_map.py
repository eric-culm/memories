#epochs selected for every srnn model
'''
models_map = which epochs are selected in the training of each model
0 = best, most clear model
1-4 = progressively degrading models

samples_map = how many samples of each duration are generated
'''

samples_map = {
                1: 300,
                3: 200,
                10: 100,
                30: 60,
                60: 60
}

#macro categories
models_map = {
                'instrumental': {},
                'fieldrec': {},
                'voice': {}
              }

#fill dict with detailed info
models_map['instrumental'] = {
                                'africanPercs' : {0:90,
                                                1:4,
                                                2:54},


                                'ambient1' : {0:20,
                                                1:19,
                                                2:27},



                                'buchla' : {0:31,
                                                1:30,
                                                2:34,
                                                3:43},


                                'buchla2' : {0:28,
                                                1:72,
                                                2:37,
                                                3:41,
                                                4:53},


                                'classical' : {0:36,
                                                1:26,
                                                2:90},


                                'classical2' : {0:15,
                                                1:22,
                                                2:86},


                                'guitarAcoustic' : {0:20,
                                                1:43,
                                                2:16,
                                                3:62,
                                                4:65},


                                'guitarBaroque' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'jazz' : {0:19,
                                                1:24,
                                                2:13,
                                                3:15,
                                                4:77},


                                'organ' : {0:82,
                                                1:85,
                                                2:26},


                                'percsWar' : {0:11,
                                                1:9,
                                                2:10,
                                                3:23},


                                'percussions' : {0:13,
                                                1:23,
                                                2:50,
                                                3:75},


                                'pianoChill' : {0:21,
                                                1:18,
                                                2:34,
                                                3:98},


                                'pianoDreamy' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'pianoSmooth' : {0:'x',
                                                1:'x',
                                                2:'x'}


                                }

models_map['fieldrec'] = {
                                'airport' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'birdsStreet' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'forest' : {0:'x',
                                                1:'x',
                                                2:'x'},



                                'library' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'mixed' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'office' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'rain' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'sea' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'train' : {0:'x',
                                                1:'x',
                                                2:'x'},


                                'wind' : {0:'x',
                                                1:'x',
                                                2:'x'}



                            }


models_map['voice'] = {}
