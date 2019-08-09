#from __future__ import print_function
import sys, os
#look at sys argv: if in crossvalidation model i/o matrices and new model filename
#are given from crossvalidation script, otherwise are normally taken from config.ini
try:
    cross_tag =  sys.argv[1]
    if cross_tag == 'crossvalidation':
        num_experiment = sys.argv[2]
        num_run = sys.argv[3]
        num_fold = sys.argv[4]
        parameters = sys.argv[5]
        model_path = sys.argv[6]
        results_path = sys.argv[7]
        output_temp_data_path = sys.argv[8]
        dataset = sys.argv[9]
        gpu_ID = int(sys.argv[10])
        num_folds = int(sys.argv[11])
        task_type = sys.argv[12]
        parameters_path = sys.argv[13]
        task_type = sys.argv[14]
        generator = eval(sys.argv[15])
        SAVE_MODEL = model_path

        print('crossvalidation mode: I/O from crossvalidation script')
        print('')
        print ('dataset: ' + dataset)
        print ('')
        print ('saving results at: ' + results_path)
        print('saving model at: ' + SAVE_MODEL + '.hdf5')
        print ('')

except IndexError:
    #test parameters
    #IF IN TEST MODE:no xvalidation, results saved as exp0
    #generator: 11865
    #nogenerator
    generator = True
    dataset = 'daic'
    architecture = 'EXAMPLE_model_regression'
    parameters = ['niente = 0']
    task_type = 'regression'
    SAVE_MODEL = '../models/prova'
    results_path = '../results/provisional'
    parameters_path = results_path + '/parameters'
    SAVE_RESULTS = results_path
    num_fold = 0
    num_exp = 0
    num_run = 0
    num_folds = 1
    gpu_ID = 1

    print ('test mode: I/O from config.ini file')
    print ('')
    print ('dataset: ' + dataset)
    print ('')
    print ('saving results at: ' + SAVE_RESULTS)
    print ('')
    print ('saving model at: ' + SAVE_MODEL + '.hdf5')
    print ('')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ID)

import loadconfig
import configparser
import json
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils import np_utils
from keras.backend import int_shape
from keras.models import load_model
from keras import regularizers
from keras import optimizers
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import define_models as choose_model
import utility_functions as uf
#import preprocessing_DAIC as pre

#np.random.seed(0)
#torch.manual_seed(0)
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#load parameters from config file only test mode
DATASET_FOLDER = cfg.get('preprocessing', 'output_folder')

predictors_name = dataset + '_predictors.npy'
target_name = dataset + '_target.npy'
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)
TARGET_LOAD = os.path.join(DATASET_FOLDER, target_name)

#default training parameters
train_split = cfg.getfloat('training_defaults', 'train_split')
validation_split = cfg.getfloat('training_defaults', 'validation_split')
test_split = cfg.getfloat('training_defaults', 'test_split')
shuffle_training_data = eval(cfg.get('training_defaults', 'shuffle_training_data'))
save_best_model_metric = cfg.get('training_defaults', 'save_best_model_metric')
save_best_model_mode = cfg.get('training_defaults', 'save_best_model_mode')
early_stopping = eval(cfg.get('training_defaults', 'early_stopping'))
patience = cfg.getint('training_defaults', 'patience')
batch_size = cfg.getint('training_defaults', 'batch_size')
num_epochs = cfg.getint('training_defaults', 'num_epochs')
learning_rate = cfg.getfloat('training_defaults', 'learning_rate')
regularization_lambda = cfg.getfloat('training_defaults', 'regularization_lambda')
optimizer = cfg.get('training_defaults', 'optimizer')


percs = [train_split, validation_split, test_split]

if task_type == 'classification':
    loss_function = 'categorical_crossentropy'
    metrics_list = ['accuracy']
elif task_type == 'regression':
    loss_function = 'MSE'
    metrics_list = ['MAE']

else:
    raise ValueError('task_type can be only: multilabel_classification, binary_classification or regression')

#path for saving best val loss and best val acc models
BVL_model_path = SAVE_MODEL + '.hdf5'

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
    for param in parameters:
        exec(param)

except IndexError:
    pass


#define optimizer ADD HERE DIFFERENT OPTIMIZERS!!!!!!!
if optimizer == 'adam':
    opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
elif optimizer == 'sgd':
    opt.optimizers.SGD(lr=learning_rate)
else:
    raise ValueError('Bad optimizer chosen')

#build dict with all UPDATED training parameters
training_parameters = {'train_split': train_split,
    'validation_split': validation_split,
    'test_split': test_split,
    'shuffle_training_data': shuffle_training_data,
    'save_best_model_metric': save_best_model_metric,
    'save_best_model_mode': save_best_model_mode,
    'early_stopping': early_stopping,
    'patience': patience,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'regularization_lambda': regularization_lambda,
    'optimizer': optimizer
    }

def main():
    #CREATE DATASET
    #load numpy data
    print('\n loading dataset...')

    folds_dataset_path = '../dataset/matrices'
    curr_fold_string = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    curr_fold_path = os.path.join(folds_dataset_path, curr_fold_string)

    train_pred_path = dataset + '_training_predictors_fold_' + str(num_fold) + '.npy'
    train_target_path = dataset + '_training_target_fold_' + str(num_fold) + '.npy'
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    train_target_path = os.path.join(folds_dataset_path, train_target_path)

    val_pred_path = dataset + '_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_target_path = dataset + '_validation_target_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    val_target_path = os.path.join(folds_dataset_path, val_target_path)

    test_pred_path = dataset + '_test_predictors_fold_' + str(num_fold) + '.npy'
    test_target_path = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)
    test_target_path = os.path.join(folds_dataset_path, test_target_path)

    #compute which actors put in train, val, test for current fold
    dummy = np.load(TARGET_LOAD)
    dummy = dummy.item()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
    #TRAIN/VAL/TEST IN A BALANCED WAY
    foldable_list = list(dummy.keys())
    fold_actors_list = uf.folds_generator(num_folds, foldable_list, percs)
    train_list = fold_actors_list[int(num_fold)]['train']
    val_list = fold_actors_list[int(num_fold)]['val']
    test_list = fold_actors_list[int(num_fold)]['test']
    del dummy

    #if tensors of current fold has not been computed:
    if not os.path.exists(test_target_path):
        #load merged dataset, compute and save current tensors
        predictors_merged = np.load(PREDICTORS_LOAD)
        target_merged = np.load(TARGET_LOAD)
        predictors_merged = predictors_merged.item()
        target_merged = target_merged.item()

        print ('\n building dataset for current fold')
        print ('\n training:')
        training_predictors, training_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, train_list)
        print ('\n validation:')

        validation_predictors, validation_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, val_list)
        print ('\n test:')
        test_predictors, test_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, test_list)

        np.save(train_pred_path, training_predictors)
        np.save(train_target_path, training_target)
        np.save(val_pred_path, validation_predictors)
        np.save(val_target_path, validation_target)
        np.save(test_pred_path, test_predictors)
        np.save(test_target_path, test_target)

    else:
        #load pre-computed tensors
        training_predictors = np.load(train_pred_path)
        training_target = np.load(train_target_path)
        validation_predictors = np.load(val_pred_path)
        validation_target = np.load(val_target_path)
        test_predictors = np.load(test_pred_path)
        test_target = np.load(test_target_path)

    #normalize to 0 mean and unity std (according to training set mean and std)
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)

    #OVERFITTING TEST!!! REMOVE THESE LINES FOR PROPER TRAINING
    '''
    validation_predictors = training_predictors.copy()
    validation_target = training_target.copy()
    '''

    #normalize labels between 0 and 1
    '''
    max_labels = [np.max(training_target), np.max(validation_target), np.max(test_target)]
    max_val = float(np.max(max_labels))
    training_target = np.divide(training_target, max_val)
    validation_target = np.divide(validation_target, max_val)
    test_target = np.divide(test_target, max_val)
    '''

    #select a subdataset for testing (to be commented when normally trained)
    '''
    bound = 30
    training_predictors = training_predictors[:bound]
    training_target = training_target[:bound]
    validation_predictors = validation_predictors[:bound]
    validation_target = validation_target[:bound]
    test_predictors = test_predictors[:bound]
    test_target = test_target[:bound]
    '''

    #reshape tensors
    #INSERT HERE FUNCTION FOR CUSTOM RESHAPING!!!!!
    if reshaping_type == 'cnn':
        training_predictors = training_predictors.reshape(training_predictors.shape[0], training_predictors.shape[1],training_predictors.shape[2], 1)
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], validation_predictors.shape[1], validation_predictors.shape[2], 1)
        test_predictors = test_predictors.reshape(test_predictors.shape[0], test_predictors.shape[1], test_predictors.shape[2], 1)
        time_dim = training_predictors.shape[1]
        features_dim = training_predictors.shape[2]
    elif reshaping_type == 'rnn':
        time_dim = training_predictors.shape[1]
        features_dim = training_predictors.shape[2]
    elif reshaping_type == 'none':
        time_dim = training_predictors.shape[1]
        features_dim = training_predictors.shape[2]
    else:
        raise ValueError('wrong reshaping type')


    #load and compile model (model is in locals()['model'])
    print('\n loading model...')
    model_string = 'model, model_parameters = choose_model.' + architecture + '(time_dim, features_dim, parameters)'
    exec(model_string)
    locals()['model'].compile(loss=loss_function, optimizer=opt, metrics=metrics_list)
    print (locals()['model'].summary())
    #print (locals()['model_parameters'])

    #callbacks
    best_model = ModelCheckpoint(SAVE_MODEL, monitor=save_best_model_metric, save_best_only=True, mode=save_best_model_mode)  #save the best model
    early_stopping_monitor = EarlyStopping(patience=patience)  #stops training when the model is not improving
    if early_stopping:
        callbacks_list = [early_stopping_monitor, best_model]
    else:
        callbacks_list = [best_model]

    #run training
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


    #if training with generator
    if generator:  #if loading one batch at time to GPU
        datagen = ImageDataGenerator()

        history = locals()['model'].fit_generator(datagen.flow(training_predictors, training_target, batch_size=batch_size,
                shuffle=shuffle_training_data), validation_data=datagen.flow(validation_predictors, validation_target, batch_size=batch_size,
                shuffle=False), validation_steps=len(validation_target)/batch_size, callbacks=callbacks_list,
                steps_per_epoch=len(training_target)/batch_size, epochs=num_epochs, shuffle=shuffle_training_data)

    else:  #if loading all dataset to GPU
        history = locals()['model'].fit(training_predictors,training_target, epochs=num_epochs,
                                validation_data=(validation_predictors,validation_target), callbacks=callbacks_list,
                                batch_size=batch_size, shuffle=shuffle_training_data)

    train_loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    if task_type == 'classification':
        train_acc_hist = history.history['acc']
        val_acc_hist = history.history['val_acc']

    #compute results on the best saved model
    K.clear_session()  #free GPU
    best_model = load_model(SAVE_MODEL)  #load best saved model

    if generator:
        #del training_generator  #delete shuffled generator
        #build non shuffled generator
        train_score = best_model.evaluate_generator(datagen.flow(training_predictors, training_target,
                        batch_size=batch_size, shuffle=False), steps=len(training_target)/batch_size)
        val_score = best_model.evaluate_generator(datagen.flow(validation_predictors, validation_target,
                        batch_size=batch_size, shuffle=False), steps=len(validation_target)/batch_size)
        test_score = best_model.evaluate_generator(datagen.flow(test_predictors, test_target,
                        batch_size=batch_size, shuffle=False), steps=len(test_target)/batch_size)

        train_pred = best_model.predict_generator(datagen.flow(training_predictors, training_target,
                        batch_size=batch_size, shuffle=False), steps=len(training_target)/batch_size)
        val_pred = best_model.predict_generator(datagen.flow(validation_predictors, validation_target,
                        batch_size=batch_size, shuffle=False), steps=len(validation_target)/batch_size)
        test_pred = best_model.predict_generator(datagen.flow(test_predictors, test_target,
                        batch_size=batch_size, shuffle=False), steps=len(test_target)/batch_size)
    else:
        train_score = best_model.evaluate(training_predictors, training_target)
        val_score = best_model.evaluate(validation_predictors, validation_target)
        test_score = best_model.evaluate(test_predictors, test_target)
        train_pred = best_model.predict(training_predictors)
        val_pred = best_model.predict(validation_predictors)
        test_pred = best_model.predict(test_predictors)


    #save results in temp dict file
    temp_results = {}

    #save loss
    temp_results['train_loss'] = train_score[0]
    temp_results['val_loss'] = val_score[0]
    temp_results['test_loss'] = test_score[0]

    #if classification compute also f1, precision, recall
    if task_type == 'classification':
        #pred
        train_pred = np.argmax(train_pred, axis=1)
        val_pred = np.argmax(val_pred, axis=1)
        test_pred = np.argmax(test_pred, axis=1)
        #precision
        train_precision = precision_score(train_pred, np.argmax(training_target, axis=1) , average="macro")
        val_precision = precision_score(val_pred, np.argmax(validation_target, axis=1) , average="macro")
        test_precision = precision_score(test_pred, np.argmax(test_target, axis=1) , average="macro")
        #recall
        train_recall = recall_score(train_pred, np.argmax(training_target, axis=1) , average="macro")
        val_recall = recall_score(val_pred, np.argmax(validation_target, axis=1) , average="macro")
        test_recall = recall_score(test_pred, np.argmax(test_target, axis=1) , average="macro")
        #f1
        train_f1 = f1_score(train_pred, np.argmax(training_target, axis=1) , average="macro")
        val_f1 = f1_score(val_pred, np.argmax(validation_target, axis=1) , average="macro")
        test_f1 = f1_score(test_pred, np.argmax(test_target, axis=1) , average="macro")

        temp_results['train_acc'] = train_score[1]
        temp_results['val_acc'] = val_score[1]
        temp_results['test_acc'] = test_score[1]

        temp_results['train_f1'] = train_f1
        temp_results['val_f1'] = val_f1
        temp_results['test_f1'] = test_f1

        temp_results['train_precision'] = train_precision
        temp_results['val_precision'] = val_precision
        temp_results['test_precision'] = test_precision

        temp_results['train_recall'] = train_recall
        temp_results['val_recall'] = val_recall
        temp_results['test_recall'] = test_recall
    #save acc if classification append classification metrics
    elif task_type == 'regression':
        temp_results['train_MAE'] = train_score[1]
        temp_results['val_MAE'] = val_score[1]
        temp_results['test_MAE'] = test_score[1]

        temp_results['train_RMSE'] = np.sqrt(train_score[0])
        temp_results['val_RMSE'] = np.sqrt(val_score[0])
        temp_results['test_RMSE'] = np.sqrt(test_score[0])

    #save history
    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = val_loss_hist
    if task_type == 'classification':
        temp_results['train_acc_hist'] = train_acc_hist
        temp_results['val_acc_hist'] = val_acc_hist

    #save actors present in current fold
    temp_results['training_actors'] = train_list
    temp_results['validation_actors'] = val_list
    temp_results['test_actors'] = test_list

    #save parameters dict
    for i in training_parameters.keys():
        if i in locals()['model_parameters'].keys():
            del locals()['model_parameters'][i]

    with open(parameters_path, 'w') as f:
        f.write('%s\n' % ('TRAINING PARAMETERS:'))
        for key, value in training_parameters.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s\n' % (''))
        f.write('%s\n' % ('MODEL PARAMETERS:'))
        for key, value in locals()['model_parameters'].items():
            f.write('%s:%s\n' % (key, value))

    np.save(results_path, temp_results)

    #print train results
    print ('')
    print ('\n train results:')
    for i in temp_results.keys():
        if 'hist' not in i:
            if 'train' in i:
                print (str(i) + ': ' + str(temp_results[i]))
    print ('\n val results:')
    for i in temp_results.keys():
        if 'hist' not in i:
            if 'val' in i:
                print (str(i) + ': ' + str(temp_results[i]))
    print ('\n test results:')
    for i in temp_results.keys():
        if 'hist' not in i:
            if 'test' in i:
                print (str(i) + ': ' + str(temp_results[i]))

if __name__ == '__main__':
    main()















#
