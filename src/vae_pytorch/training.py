from __future__ import print_function
import sys, os
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


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#global parameters from config file
dataset = cfg.get('vae', 'train_input_dataset')
experiment_name = cfg.get('vae', 'experiment_name')
architecture = cfg.get('vae', 'architecture')
output_models_path = cfg.get('vae', 'output_models_path')
load_pretrained = eval(cfg.get('vae', 'load_pretrained'))
pretrained_path = cfg.get('vae', 'pretrained_path')
gpu_ID = cfg.getint('vae', 'gpu_id')
cuda = eval(cfg.get('vae', 'cuda'))
batch_size = cfg.getint('vae', 'batch_size')
results_path = cfg.get('vae', 'output_results_path')

SAVE_MODEL = os.path.join(output_models_path, experiment_name)

#default training parameters
#globals
verbose = False
model_size = 64
variational = True
beta = 2.
latent_dim = 10
clip_gradients = 1.
dropout = False
patience = 10
num_epochs = 800000
learning_rate = 0.0001
regularization_lambda = 0.
optimizer = 'adam'
features_type = 'waveform'

#dataset division
train_split = 0.8
validation_split = 0.1
test_split = 0.1
subdataset_bound = 0
offset_bound = 0
shuffle_training_data = True

#saving\
training_dict_path = os.path.join(results_path, 'stats')
save_best_only = False
save_model_xepochs = True
save_model_nepochs = 100
save_latent_distribution = True
save_distribution_epochs_n = 100
save_figs = False
save_sounds = False
save_items_epochs = 100
save_items_n = 1

#warm ups
convergence_threshold = 0.1
warm_up = True
kld_holes = True
kld_epochs_n = 3
warm_up_after_convergence = False
warm_up_kld = False
warm_up_reparametrize = False
kld_ramp_delay = 30
kld_ramp_epochs = 1500
reparametrize_ramp_delay = 30
reparametrize_ramp_epochs = 1500

percs = [train_split, validation_split, test_split]

if cuda:
    device = torch.device('cuda:' + str(gpu_ID))
else:
    device = torch.device('cpu')

#build dict with model PARAMETERS
parameters = {}
parameters['verbose'] = verbose
parameters['model_size'] = model_size
parameters['variational'] = variational
parameters['beta'] = beta
parameters['warm_up'] = warm_up
parameters['latent_dim'] = latent_dim
parameters['clip_gradients'] = clip_gradients
parameters['dropout'] = dropout
parameters['regularization_lambda'] = regularization_lambda

#training routine
def main():
    #create results path
    gen_figs_path = os.path.join(results_path, 'gen_figs')
    if not os.path.exists(gen_figs_path):
        os.makedirs(gen_figs_path)
    gen_sounds_path = gen_figs_path
    #load data and split into train, validation and test
    data = np.load(dataset, allow_pickle=True)
    np.random.shuffle(data) #shuffle order
    np.random.shuffle(data)
    num_data = data.shape[0]
    tr_bound = int(num_data * train_split)
    val_bound = int(num_data * (train_split + validation_split))
    training_predictors = data[:tr_bound]
    validation_predictors = data[tr_bound:val_bound]
    test_predictors = data[val_bound:]


    #reshape tensors
    training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1])
    validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1])
    test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1])


    #select a subdataset for testing (to be commented when normally trained)

    if subdataset_bound != 0:
        training_predictors = training_predictors[offset_bound:offset_bound+subdataset_bound]
        validation_predictors = validation_predictors[:subdataset_bound]
        test_predictors = test_predictors[:subdataset_bound]


    print ('Training predictors shape: ' + str(training_predictors.shape))

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float()
    val_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()


    #build dataset from tensors
    #target i == predictors because autoencoding
    tr_dataset = utils.TensorDataset(train_predictors, train_predictors)
    val_dataset = utils.TensorDataset(val_predictors, val_predictors)
    test_dataset = utils.TensorDataset(test_predictors, test_predictors)

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=shuffle_training_data, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=shuffle_training_data, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=shuffle_training_data, pin_memory=True)  #no batch here!!
    #DNN input shape
    parameters['input_dim'] = train_predictors.shape[-1]



    #load model (model is in locals()['model'])
    print('\nloading models...')

    model_string = 'model_class, model_parameters = choose_model.' + architecture + '(parameters)'
    exec(model_string)
    model = locals()['model_class'].to(device)

    #load pretrained if specified
    if load_pretrained:
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    #create results folders
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

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
        warm_ramp_kld = training_utils.warm_up_kld(num_epochs, kld_ramp_delay, kld_ramp_epochs)
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

        if warm_up_kld:
            warm_value_kld = warm_value_kld
        else:
            warm_value_kld = 1.

        if warm_up_reparametrize:
            warm_value_reparametrize = warm_value_reparametrize
        else:
            warm_value_reparametrize = 1.




        print ('\n')
        string = 'Epoch: [' + str(epoch+1) + '/' + str(num_epochs) + '] '
        #iterate batches

        model.train()
        for i, (sounds, truth) in enumerate(tr_data):
                sounds = sounds.to(device)
                optimizer_joint.zero_grad()

                outputs, mu, logvar = model(sounds, warm_value_reparametrize)

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
                if clip_gradients is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
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

                    outputs, mu, logvar = model(sounds, warm_value_reparametrize)

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

                    outputs, mu, logvar = model(sounds, warm_value_reparametrize)

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
                    'training', warm_value_reparametrize, gen_distributions_path, save_latent_distribution)
            #test_data
            uf.save_data(test_data, model, device, epoch, gen_figs_path, gen_sounds_path,
                    save_figs, save_sounds, save_items_epochs, save_items_n, features_type,
                    'test', warm_value_reparametrize, gen_distributions_path, save_latent_distribution)

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

            print('convrgence flag: ' + str(convergence_flag) + ' | loss worm: ' +
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





if __name__ == '__main__':
    main()















#
