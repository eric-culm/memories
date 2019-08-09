#GENERAL DESCRIPTION
This API is aimed at asily defining and running experiments, intended as routines
of trainings. In each instance of an experiment it is possible to specify any kind
parameter, as well as automatically performing k-fold cross-validation. The outcomes
of an experiment are saved in a custom-defined folder, which contains:
  -A dict with all metrics and history, separately computed for every instance
    of the experiment and for every k-fold.
  -All models (.hdf5), separately computed for every instance of the experiment
    and for every k-fold.
  -A copy of he current version of the code, saved at the moment of running of the
  last instance of the experiment.
  -A txt file containing a short description of what has been done in the experiment
  -A spreadsheet that shows the most important metrics and highlights the best
    results.


#SCRIPTS
-xval_routine: UI to define an experiment and its instances. This script iterates
  all instances of an experiment calling the script xval_instance.
-xval_instance: This script automatically performs the k-fold cross-validation. It
  iterates every fold calling the script build_model.
-build_model: This script runs the actual trainings.
-models: in this script it is possible to define custom keras models.
-utility_functions: self-explanatory.
-results_to_excel: computes the spreadsheets.
-preprocessing_DATASET: processes audio data building the features matrices calling
  the script feat_analysis.
-feat_analysis: contains the feature extraction functions: STFT and MFCC
-config.ini: This config file contains mainly I/O folder paths


#EXPERIMENT DEFINITION
For each experiment you can create a new xval_routine script, copying the example one.
In each experiment it is mandatory to define these macro parameters:
  -A short description of the experiment that will be saved in a txt file. For
    example 'In this experiment we tested different learning rates'
  -dataset: a short name of the used dataset. This will affect the name of the
    results and serves as well to load the correct dataset.
  -num_experiment: number of current experiment (has to be an integer).
  -num_folds: int, how many k for the k-fold cross-validation.
  -experiment_folder: path in which save all results. Different
    experiments for the same dataset are saved in the same directory.
In each experiment you should define a dict containing the instances of the experiment.
The keys should be progressive integers.
Each key/instance has to be a list of strings and each element of a list is a
parameter declaration.
Example:
experiment_dict[1] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg base 0.001"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.001"']
experiment_dict[2] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg increased 0.01"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.01"']

The parameters you insert overwrite the default one, which are declared in the
build_model and models_API scripts. Since a copy of the code issaved for every
experiment, you can easily check which were the default parameters in case you change them.
In each instance it is mandatory to declare at least these 4 parameters (See previous example):
  -comment_1 and comment_2: comments that are plotted in the spreadsheet.
  -task_type: should be 'classification' or 'regression'
  -architecture: the model you want to use. Should be the name of a model function
    present in models_API script.

#CUSTOM MODELS DEFINITION
To define a model follow the instructions written in models_API.EXAMPLE_model()


#PREPROCESSING
Unfortunately, the preprocessing needs to be customized for every new dataset.
In order to be compatible con the rest of the API, any proprocessing script has to
output 2 dictionaries: 1 containing the predictors and 1 containing the target.
The keys of these dicts has to be the 'foldable items', that is the criterion
that you want to use to divide train-va-test, for example the different actors in
a speech dataset. So every key should contain all data from one single actor.
example:
predictors_dict['1': matrix with all spectra of actor 1,
                '2': matrix with all spectra of actor 2]
target_dict['1': matrix with all labels of actor 1,
            '2': matrix with all labels of actor 1]
