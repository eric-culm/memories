#epochs selected for every srnn model
'''
models_map = which epochs are selected in the training of each model
0 = best, most clear model
1-4 = progressively degrading models

samples_map = how many samples of each duration are generated
'''

durations_map = {
                3: 200,
                5: 200,
                10: 100,
                30: 100,
                60: 80
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


                                'guitarBaroque' : {0:18,
                                                1:70,
                                                2:39,
                                                3:48,
                                                4:99},


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


                                'pianoDreamy' : {0:11,
                                                1:20,
                                                2:27,
                                                3:8,
                                                4:61,
                                                5:75},


                                'pianoSmooth' : {0:18,
                                                1:22,
                                                2:28,
                                                3:30,
                                                4:37}


                                }

models_map['fieldrec'] = {
                                'airport' : {0:20,
                                                1:34,
                                                2:12},


                                'birdsStreet' : {0:18,
                                                1:14,
                                                2:31},


                                'forest' : {0:32,
                                                1:36,
                                                2:67},



                                'library' : {0:56,
                                                1:14,
                                                2:29},


                                'mixed' : {0:15,
                                                1:18,
                                                2:21},


                                'office' : {0:50,
                                                1:48,
                                                2:58},


                                'rain' : {0:31,
                                                1:45,
                                                2:59},


                                'sea' : {0:13,
                                                1:15,
                                                2:45},


                                'train' : {0:22,
                                                1:16,
                                                2:54},


                                'wind' : {0:10,
                                                1:19,
                                                2:37}


                            }


models_map['voice'] = {}
