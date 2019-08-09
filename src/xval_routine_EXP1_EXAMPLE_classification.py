from __future__ import print_function
import numpy as np
import sys, os
import xval_instance as xval

#EXPERIMENT PARAMETERS:
gpu_ID = 0
overwrite_results = True  #if true overwrite existing experiment instances
debug_mode = True  #if false, if an error occurs in one instance, it is skipped without stopping the routine
short_description = 'stupid test'
dataset = 'ravdess'
task_type = 'classification'
generator = False
num_experiment = 1  #id of the experiment
num_folds = 2  #number of k-folds for cross-validation
#experiment_folder = '../../../copy/prova_API'  #where to save results
experiment_folder = '../results/digits'  #where to save results

#DEFINE HERE EVERY INSTANCE OF THE EXPERIMENT
#every instance must be a key in the experiment dict
#every key must be a list of strings
#every parameter overwrites the correspective default parameter
#mandatory parameters:
#-task_type: classification, or regression
#-reshaping type: conv, lstm, none
#-architecture: one of the models defined in the models_API script
#-comment_1 and comment_2: write here any info you want to show in the results spreadsheet. Example: L2 increased to 0.1
experiment = {}
experiment[1] = ['architecture="EXAMPLE_model_classification"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.001"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.001"']
experiment[2] = ['architecture="EXAMPLE_model_classification"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.01"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.01"']
experiment[3] = ['architecture="EXAMPLE_model_classification"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.1"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.1"']
experiment[4] = ['architecture="EXAMPLE_model_classification"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.3"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.3"']


#DON'T TOUCH WHAT IS WRITTEN BELOW THIS LINE
#-------------------------------------------------------------------------------#
print ('Overwrite results: ' + str(overwrite_results))
print ('Debug mode: ' + str(debug_mode))

output_path = experiment_folder + '/experiment_' + str(num_experiment)
if not os.path.exists(output_path):
    os.makedirs(output_path)
description_path = output_path + '/experiment_' + str(num_experiment) + '_description.txt'

with open(description_path, "w") as text_file:
    text_file.write(short_description)


try:
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    gpu_ID = int(sys.argv[3])

except IndexError:
    #if not specified run all experiments
    keys = list(experiment.keys())
    begin = keys[0]
    end = keys[-1]

for num_run in range(begin,end+1):
    results_name = output_path + '/results/results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    temp_params = '/'.join(experiment[num_run])

    if overwrite_results:
        if debug_mode == False:
            try:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
            except:
                pass
        else:
            xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)

    else:  #if not overwrite results
        if not os.path.exists(output_path):  #not overwrite experiments
            if debug_mode == False:
                try:
                    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
                except:
                    pass
            else:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
        else:  #if result exists print below line
            print ('exp' + str(num_experiment) + ' run' + str(num_run) + ' already exists: skipping')
print('')
print ('REQUESTED EXPERIMENTS SUCCESSFULLY COMPLETED')
