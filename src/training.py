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
        print('saving model at: ' + SAVE_MODEL)
        print ('')

except IndexError:
    #test parameters
    #IF IN TEST MODE:no xvalidation, results saved as exp0
    #generator: 11865
    #nogenerator
    dataset = 'sc09_1000s_waveform'
    exp_name = 'sc09_nowaitCOnverg'

    architecture = 'WAVE_CNN_complete_net'
    parameters = ['verbose=False', 'model_size=64', 'variational=True',
                  'beta=1.', 'warm_up=True', 'latent_dim=100',
                  'subdataset_bound=100',
                  'features_type="waveform"']

    SAVE_MODEL = '../models/' + exp_name
    results_path = '../results/' + exp_name
    training_dict_path = '../results/' + exp_name + '_training_dict.npy'
    parameters_path = results_path + '/parameters'
    SAVE_RESULTS = results_path
    num_fold = 0
    num_experiment = 0
    num_run = 0
    num_folds = 1
    gpu_ID = 1


    print ('test mode: I/O from config.ini file')
    print ('')
    print ('dataset: ' + dataset)
    print ('')
    print ('saving results at: ' + SAVE_RESULTS)
    print ('')
    print ('saving model at: ' + SAVE_MODEL)
    print ('')


import loadconfig
import configparser
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from audtorch import metrics
import numpy as np
import define_models as choose_model
import utility_functions as uf
import training_utils
#import preprocessing_DAIC as pre

#np.random.seed(0)
#torch.manual_seed(0)
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)


#default training parameters
train_split = cfg.getfloat('training_defaults', 'train_split')
validation_split = cfg.getfloat('training_defaults', 'validation_split')
test_split = cfg.getfloat('training_defaults', 'test_split')
shuffle_training_data = eval(cfg.get('training_defaults', 'shuffle_training_data'))
patience = cfg.getint('training_defaults', 'patience')
batch_size = cfg.getint('training_defaults', 'batch_size')
num_epochs = cfg.getint('training_defaults', 'num_epochs')
learning_rate = cfg.getfloat('training_defaults', 'learning_rate')
regularization_lambda = cfg.getfloat('training_defaults', 'regularization_lambda')
optimizer = cfg.get('training_defaults', 'optimizer')
recompute_matrices = eval(cfg.get('training_defaults', 'recompute_matrices'))
convergence_threshold = cfg.getfloat('training_defaults', 'convergence_threshold')
load_pretrained = eval(cfg.get('training_defaults', 'load_pretrained'))
pretrained_path = cfg.get('training_defaults', 'pretrained_path')


save_best_only = eval(cfg.get('training_defaults', 'save_best_only'))
save_model_xepochs = eval(cfg.get('training_defaults', 'save_model_xepochs'))
save_model_nepochs = eval(cfg.get('training_defaults', 'save_model_nepochs'))
save_model_freq = eval(cfg.get('training_defaults', 'save_model_freq'))
save_model_epochs = cfg.getint('training_defaults', 'save_model_epochs')
save_sounds = eval(cfg.get('training_defaults', 'save_sounds'))
save_figs = eval(cfg.get('training_defaults', 'save_figs'))
save_items_epochs = cfg.getint('training_defaults', 'save_items_epochs')
save_items_n = cfg.getint('training_defaults', 'save_items_n')
save_latent_distribution = eval(cfg.get('training_defaults', 'save_latent_distribution'))
save_distribution_epochs_n = cfg.getint('training_defaults', 'save_distribution_epochs_n')

kld_holes = eval(cfg.get('training_defaults', 'kld_holes'))
kld_epochs_n = cfg.getint('training_defaults', 'kld_epochs_n')
warm_up_after_convergence = eval(cfg.get('training_defaults', 'warm_up_after_convergence'))
warm_up_kld = eval(cfg.get('training_defaults', 'warm_up_kld'))
warm_up_reparametrize = eval(cfg.get('training_defaults', 'warm_up_reparametrize'))
kld_ramp_delay = cfg.getint('training_defaults', 'kld_ramp_delay')
kld_ramp_epochs = cfg.getint('training_defaults', 'kld_ramp_epochs')
reparametrize_ramp_delay = cfg.getint('training_defaults', 'reparametrize_ramp_delay')
reparametrize_ramp_epochs = cfg.getint('training_defaults', 'reparametrize_ramp_epochs')

percs = [train_split, validation_split, test_split]

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
except IndexError:
    pass
for param in parameters:
    exec(param)

#load folder parameters from config.ini
DATASET_FOLDER = cfg.get('preprocessing', 'output_folder')
SR = cfg.getint('sampling', 'sr_target')
predictors_name = dataset + '_predictors.npy'
target_name = dataset + '_target.npy'
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)
TARGET_LOAD = os.path.join(DATASET_FOLDER, target_name)

device = torch.device('cuda:' + str(gpu_ID))

#build dict with all UPDATED training parameters
training_parameters = {'train_split': train_split,
    'validation_split': validation_split,
    'test_split': test_split,
    'shuffle_training_data': shuffle_training_data,
    'patience': patience,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'regularization_lambda': regularization_lambda,
    'optimizer': optimizer,
    'warm_up_kld': warm_up_kld,
    'warm_up_reparametrize': warm_up_reparametrize,
    'kld_ramp_delay': kld_ramp_delay,
    'kld_ramp_epochs': kld_ramp_epochs,
    'reparametrize_ramp_delay': reparametrize_ramp_delay,
    'reparametrize_ramp_epochs': reparametrize_ramp_epochs,
    'save_model_freq': save_model_freq,
    'warm_up_after_convergence': warm_up_after_convergence,
    'recompute_matrices': recompute_matrices,
    'convergence_threshold': convergence_threshold,
    'load_pretrained': load_pretrained,
    'pretrained_path': pretrained_path,
    'kld_epochs_n': kld_epochs_n,
    'kld_holes': kld_holes
    }


def main():
    #CREATE DATASET
    #load numpy data
    print('\n loading dataset...')

    folds_dataset_path = '../dataset/matrices'
    curr_fold_string = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    curr_fold_path = os.path.join(folds_dataset_path, curr_fold_string)

    train_pred_path = dataset + '_training_predictors_fold_' + str(num_fold) + '.npy'
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    val_pred_path = dataset + '_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    test_pred_path = dataset + '_test_predictors_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)

    train_target_path = dataset + '_training_target_fold_' + str(num_fold) + '.npy'
    train_target_path = os.path.join(folds_dataset_path, train_target_path)
    val_target_path = dataset + '_validation_target_fold_' + str(num_fold) + '.npy'
    val_target_path = os.path.join(folds_dataset_path, val_target_path)
    test_target_path = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    test_target_path = os.path.join(folds_dataset_path, test_target_path)


    '''
    training_predictors, validation_predictors, test_predictors = uf.get_dataset_matrices(
                data_path=PREDICTORS_LOAD, num_folds=num_folds, num_fold=num_fold,
                percs=percs, train_path=train_pred_path, val_path=val_pred_path,
                test_path=test_pred_path, recompute_matrices=recompute_matrices)
    '''

    #compute which actors put in train, val, test for current fold
    dummy = np.load(TARGET_LOAD,allow_pickle=True)
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
    if recompute_matrices:
        predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
        target_merged = np.load(TARGET_LOAD,allow_pickle=True)
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

    if not recompute_matrices:
        if not os.path.exists(test_target_path):
            #load merged dataset, compute and save current tensors
            predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
            target_merged = np.load(TARGET_LOAD,allow_pickle=True)
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
            training_predictors = np.load(train_pred_path,allow_pickle=True)
            training_target = np.load(train_target_path,allow_pickle=True)
            validation_predictors = np.load(val_pred_path,allow_pickle=True)
            validation_target = np.load(val_target_path,allow_pickle=True)
            test_predictors = np.load(test_pred_path,allow_pickle=True)
            test_target = np.load(test_target_path,allow_pickle=True)


    #normalize to 0-1
    tr_max = np.max(training_predictors)
    training_predictors = np.divide(training_predictors, tr_max)
    validation_predictors = np.divide(validation_predictors, tr_max)
    test_predictors = np.divide(test_predictors, tr_max)

    #reshape tensors
    if features_type == 'waveform':
        training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[2])
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[2])
        test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[2])
    elif features_type =='spectrum':
        training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
        test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

    '''
    training_target = training_predictors
    validation_target = validation_predictors
    test_target = test_predictors
    '''


    #select a subdataset for testing (to be commented when normally trained)
    if subdataset_bound != 'all':
        training_predictors = training_predictors[:subdataset_bound]
        validation_predictors = validation_predictors[:subdataset_bound]
        test_predictors = test_predictors[:subdataset_bound]
        training_target = training_target[:subdataset_bound]
        validation_target = validation_target[:subdataset_bound]
        test_target = test_target[:subdataset_bound]

    print ('Training predictors shape: ' + str(training_predictors.shape))
    print ('Training target shape: ' + str(training_target.shape))

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float()
    val_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    train_target = torch.tensor(training_target).float()
    val_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()

    #build dataset from tensors
    #target i == predictors because autoencoding
    tr_dataset = utils.TensorDataset(train_predictors,train_target)
    val_dataset = utils.TensorDataset(val_predictors, val_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=shuffle_training_data, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)  #no batch here!!
    #DNN input shape
    time_dim = training_predictors.shape[-2]
    features_dim = training_predictors.shape[-1]


    #load model (model is in locals()['model'])
    print('\n loading models...')

    model_string = 'model_class, model_parameters = choose_model.' + architecture + '(time_dim, features_dim, parameters)'
    exec(model_string)
    model = locals()['model_class'].to(device)

    #load pretrained if specified
    if load_pretrained:
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    #create results folders
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    gen_sounds_path = os.path.join(results_path, 'gen_sounds')
    if not os.path.exists(gen_sounds_path):
        os.makedirs(gen_sounds_path)
    gen_figs_path = os.path.join(results_path, 'gen_figs')
    if not os.path.exists(gen_figs_path):
        os.makedirs(gen_figs_path)
    gen_distributions_path = os.path.join(results_path, 'hidden_distribution')
    if not os.path.exists(gen_distributions_path):
        os.makedirs(gen_distributions_path)

    #compute number of parameters
    print ('')
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #define optimizer
    optimizer_joint = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=regularization_lambda)

    #create warm up ramps
    if not warm_up_after_convergence:
        warm_ramp_kld = training_utils.warm_up(num_epochs, kld_ramp_delay, kld_ramp_epochs)
        warm_ramp_reparametrize = training_utils.warm_up_reparametrize(num_epochs, reparametrize_ramp_delay, reparametrize_ramp_epochs)

    #init utility lists
    total_step = len(tr_data)
    loss_list = []
    train_joint_hist = []
    train_kld_hist = []
    train_recon_hist = []
    val_joint_hist = []
    val_kld_hist = []
    val_recon_hist = []
    patience_vec = []

    #init variables for dynamic warm up
    dyn_variational = False
    convergence_flag = False

    training_dict = {'train_joint_loss': [],
                     'train_recon_loss': [],
                     'train_kld_loss': [],
                     'val_joint_loss': [],
                     'val_recon_loss': [],
                     'val_kld_loss': [],
                     'variational_active': [],
                     'loss_worm': [],
                     'noise_worm': []
                     }

    #TRAINING LOOP
    #iterate epochs
    for epoch in range(num_epochs):

        if not convergence_flag:
            SAVE_MODEL_final_path = SAVE_MODEL + '_before_convergence'
        else:
            SAVE_MODEL_final_path = SAVE_MODEL + '_after_convergence'

        if warm_up_after_convergence:
            #if it is not still converged, create ramps starting
            #from curent epoch
            if not convergence_flag:
                warm_ramp_kld = training_utils.warm_up_kld(num_epochs, epoch, kld_ramp_epochs)
                warm_ramp_reparametrize = training_utils.warm_up_reparametrize(num_epochs, epoch, reparametrize_ramp_epochs)
                warm_value_kld = 0.
                warm_value_reparametrize = 0.
            #if it converged, keep ramp of last epoch
            else:
                dyn_variational = True
                warm_value_kld = warm_ramp_kld[epoch]
                warm_value_reparametrize = warm_ramp_reparametrize[epoch]

        else:
            warm_value_kld = warm_ramp_kld[epoch]
            warm_value_reparametrize = warm_ramp_reparametrize[epoch]

        #if use kld loss every k epochs
        if kld_holes:
            if epoch % kld_epochs_n == 0:
                warm_value_kld = warm_value_kld
            else:
                warm_value_kld = 0.



        print ('\n')
        string = 'Epoch: [' + str(epoch+1) + '/' + str(num_epochs) + '] '
        #iterate batches

        model.train()
        for i, (sounds, truth) in enumerate(tr_data):
                sounds = sounds.to(device)
                truth = truth.to(device)
                optimizer_joint.zero_grad()

                outputs, mu, logvar = model(sounds, dyn_variational, warm_value_reparametrize)

                loss_k = training_utils.loss_KLD(mu, logvar, warm_value_kld, outputs)
                loss_r = training_utils.loss_recon(outputs, sounds, features_type)
                loss_j = training_utils.loss_joint(outputs, sounds, mu, logvar, warm_value_kld, features_type)

                loss_j.backward(retain_graph=True)

                #print progress and update history, optimizer step
                perc = int(i / len(tr_data) * 20)
                inv_perc = int(20 - perc - 1)

                loss_k_print_t = str(np.round(loss_k.item(), decimals=5))
                loss_r_print_t = str(np.round(loss_r.item(), decimals=5))
                loss_j_print_t = str(np.round(loss_j.item(), decimals=5))

                string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_j_print_t  + ' | KLD: ' + loss_k_print_t + ' | CCC: ' + loss_r_print_t
                print ('\r', string_progress, end='')

                optimizer_joint.step()

        #validation loss, training and val accuracy computation
        #after current epoch training
        train_batch_losses_k = []
        val_batch_losses_k = []
        train_batch_losses_r = []
        val_batch_losses_r = []
        train_batch_losses_j = []
        val_batch_losses_j = []

        model.eval()
        with torch.no_grad():
            #compute training losses
            for i, (sounds, truth) in enumerate(tr_data):
                    optimizer_joint.zero_grad()
                    sounds = sounds.to(device)
                    truth = truth.to(device)

                    outputs, mu, logvar = model(sounds, dyn_variational, warm_value_reparametrize)

                    loss_k = training_utils.loss_KLD(mu, logvar, warm_value_kld, outputs, beta)
                    loss_r = training_utils.loss_recon(outputs, sounds, features_type)
                    loss_j = training_utils.loss_joint(outputs, sounds, mu, logvar, warm_value_kld, features_type, beta)

                    train_batch_losses_k.append(loss_k.item())
                    train_batch_losses_r.append(loss_r.item())
                    train_batch_losses_j.append(loss_j.item())

            #compute validation losses
            for i, (sounds, truth) in enumerate(val_data):
                    optimizer_joint.zero_grad()
                    sounds = sounds.to(device)
                    truth = truth.to(device)

                    outputs, mu, logvar = model(sounds, dyn_variational, warm_value_reparametrize)

                    loss_k = training_utils.loss_KLD(mu, logvar, warm_value_kld, outputs, beta)
                    loss_r = training_utils.loss_recon(outputs, sounds, features_type)
                    loss_j = training_utils.loss_joint(outputs, sounds, mu, logvar, warm_value_kld, features_type, beta)

                    val_batch_losses_k.append(loss_k.item())
                    val_batch_losses_r.append(loss_r.item())
                    val_batch_losses_j.append(loss_j.item())

            #average batch losses
            train_epoch_kld = np.mean(train_batch_losses_k)
            train_epoch_recon = np.mean(train_batch_losses_r)
            train_epoch_joint = np.mean(train_batch_losses_j)
            val_epoch_kld = np.mean(val_batch_losses_k)
            val_epoch_recon = np.mean(val_batch_losses_r)
            val_epoch_joint = np.mean(val_batch_losses_j)

            #append losses to history
            train_joint_hist.append(train_epoch_joint)
            train_kld_hist.append(train_epoch_kld)
            train_recon_hist.append(train_epoch_recon)
            val_joint_hist.append(val_epoch_joint)
            val_kld_hist.append(val_epoch_kld)
            val_recon_hist.append(val_epoch_recon)

            print ('\n  train_joint: ' + str(np.round(train_epoch_joint.item(), decimals=5)) + ' | val_joint: ' + str(np.round(val_epoch_joint.item(), decimals=5)))
            print ('  train_KLD: ' + str(np.round(train_epoch_kld.item(), decimals=5)) + ' | val_KLD: ' + str(np.round(val_epoch_kld.item(), decimals=5)))
            print ('  train_recon :' + str(np.round(train_epoch_recon.item(), decimals=5)) + ' | val_recon: ' + str(np.round(val_epoch_recon.item(), decimals=5)))

            #save figures if specified
            #train_data
            uf.save_data(tr_data, model, device, epoch, gen_figs_path, gen_sounds_path,
                    save_figs, save_sounds, save_items_epochs, save_items_n, features_type,
                    'training', dyn_variational, warm_value_reparametrize, gen_distributions_path, save_latent_distribution)
            #test_data
            uf.save_data(test_data, model, device, epoch, gen_figs_path, gen_sounds_path,
                    save_figs, save_sounds, save_items_epochs, save_items_n, features_type,
                    'test', dyn_variational, warm_value_reparametrize, gen_distributions_path, save_latent_distribution)

            #save best model
            if save_best_only == True:
                if epoch == 0:
                    torch.save(model.state_dict(), SAVE_MODEL_final_path)
                    print ('saved')
                    saved_epoch = epoch + 1
                else:
                    best_loss = min(train_joint_hist[:-1])  #not looking at curr_loss
                    curr_loss = train_joint_hist[-1]
                    if curr_loss < best_loss:
                        torch.save(model.state_dict(), SAVE_MODEL_final_path)
                        print ('saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                        saved_epoch = epoch + 1

            #save model every x epochs
            if save_model_xepochs == True:
                if epoch % save_model_nepochs == 0:
                    torch.save(model.state_dict(), SAVE_MODEL_final_path)
                    print ('\nModel saved')


            #update the break point if training loss is better than
            #add_THRESHOLD
            #compute mean of last 10 losses
            last_losses = train_recon_hist[-10:]
            last_mean = np.mean(last_losses)

            if last_mean <= convergence_threshold:
                convergence_flag = True

            print('  variational active: ' + str(convergence_flag) + ' | loss worm: ' +
                str(warm_value_kld) + ' | noise worm: ' + str(warm_value_reparametrize))


            #save training stats dict
            training_dict['train_joint_loss'].append(train_epoch_joint)
            training_dict['train_kld_loss'].append(train_epoch_kld)
            training_dict['train_recon_loss'].append(train_epoch_recon)
            training_dict['val_joint_loss'].append(val_epoch_joint)
            training_dict['val_kld_loss'].append(val_epoch_kld)
            training_dict['val_recon_loss'].append(val_epoch_recon)
            training_dict['loss_worm'].append(warm_value_kld)
            if convergence_flag:
                training_dict['variational_active'].append(1.)
            else:
                training_dict['variational_active'].append(0.)

            if save_model_xepochs == True:
                if epoch % save_model_nepochs == 0:
                    np.save(training_dict_path, training_dict)



            #end of epoch loop
    '''
    #compute train, val and test accuracy LOADING the best saved model
    #best validation loss
    #init batch results
    train_batch_losses = []
    val_batch_losses = []
    test_batch_losses = []


    model.load_state_dict(torch.load(BVL_model_path), strict=False)
    model.eval()
    with torch.no_grad():
        #train acc
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            temp_pred, mu, logvar = model(sounds)
            temp_loss = loss_function(temp_pred, truth, mu, logvar)
            train_batch_losses.append(temp_loss)
        #val acc
        for i, (sounds, truth) in enumerate(val_data):
            optimizer.zero_grad()
            temp_pred, mu, logvar = model(sounds)
            temp_loss = loss_function(temp_pred, truth, mu, logvar)
            val_batch_losses.append(temp_loss)
        #test acc
        for i, (sounds, truth) in enumerate(test_data):
            optimizer.zero_grad()
            temp_pred, mu, logvar = model(sounds)
            temp_loss = loss_function(temp_pred, truth, mu, logvar)
            test_batch_losses.append(temp_loss)


    #compute rounded mean of losses
    train_loss = torch.mean(torch.tensor(train_batch_losses)).cpu().numpy()
    val_loss = torch.mean(torch.tensor(val_batch_losses)).cpu().numpy()
    test_loss = torch.mean(torch.tensor(test_batch_losses)).cpu().numpy()


    #print results COMPUTED ON THE BEST SAVED MODEL
    print('')
    print ('train loss: ' + str(train_loss))
    print ('val loss: ' + str(val_loss))
    print ('test loss: ' + str(test_loss))


    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #save results in temp dict file
    temp_results = {}

    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = val_loss_hist

    temp_results['train_loss'] = train_loss
    temp_results['val_loss'] = val_loss
    temp_results['test_loss'] = test_loss

    np.save(results_path, temp_results)

    '''

if __name__ == '__main__':
    main()















#
