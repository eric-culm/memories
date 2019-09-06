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
    dataset = 'sc09_stft_hybrid'
    mnist_test = True
    architecture = 'WAVE_VAE'
    encoder_architecture = 'MNIST_encoder'
    decoder_architecture = 'WAVE_decoder'
    reparametrize_architecture = 'reparametrize'
    parameters = ['verbose=False', 'model_size=64', 'variational=True',
                  'kld_weight=-0.5', 'warm_up=True', 'latent_dim=100',
                  'hybrid_dataset=True', 'subdataset_bound=5']

    SAVE_MODEL = '../models/prova'
    results_path = '../results/provisional'
    parameters_path = results_path + '/parameters'
    SAVE_RESULTS = results_path
    num_fold = 0
    num_experiment = 0
    num_run = 0
    num_folds = 1
    gpu_ID = 0


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
from audtorch import metrics
import numpy as np
import define_models as choose_model
import utility_functions as uf
import losses
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
save_best_model_metric = cfg.get('training_defaults', 'save_best_model_metric')
save_best_model_mode = cfg.get('training_defaults', 'save_best_model_mode')
early_stopping = eval(cfg.get('training_defaults', 'early_stopping'))
patience = cfg.getint('training_defaults', 'patience')
batch_size = cfg.getint('training_defaults', 'batch_size')
num_epochs = cfg.getint('training_defaults', 'num_epochs')
learning_rate = cfg.getfloat('training_defaults', 'learning_rate')
regularization_lambda = cfg.getfloat('training_defaults', 'regularization_lambda')
optimizer = cfg.get('training_defaults', 'optimizer')
save_best_only = eval(cfg.get('training_defaults', 'save_best_only'))
save_sounds = eval(cfg.get('training_defaults', 'save_sounds'))
save_sounds_epochs = cfg.getint('training_defaults', 'save_sounds_epochs')
save_sounds_n = cfg.getint('training_defaults', 'save_sounds_n')

percs = [train_split, validation_split, test_split]

#path for saving best val loss and best val acc models
BVL_model_path = SAVE_MODEL
recompute_matrices = False

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
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)
if hybrid_dataset:
    target_name = dataset + '_target.npy'
    TARGET_LOAD = os.path.join(DATASET_FOLDER, target_name)




device = torch.device('cuda:' + str(gpu_ID))
#device = torch.device('cuda:0')


#define optimizer ADD HERE DIFFERENT OPTIMIZERS!!!!!!!

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
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    val_pred_path = dataset + '_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    test_pred_path = dataset + '_test_predictors_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)

    training_predictors, validation_predictors, test_predictors = uf.get_dataset_matrices(
                data_path=PREDICTORS_LOAD, num_folds=num_folds, num_fold=num_fold,
                percs=percs, train_path=train_pred_path, val_path=val_pred_path,
                test_path=test_pred_path, recompute_matrices=recompute_matrices)

    if hybrid_dataset:
        #load features as predictors and waveform as target
        train_target_path = dataset + '_training_target_fold_' + str(num_fold) + '.npy'
        train_target_path = os.path.join(folds_dataset_path, train_target_path)
        val_target_path = dataset + '_validation_target_fold_' + str(num_fold) + '.npy'
        val_target_path = os.path.join(folds_dataset_path, val_target_path)
        test_target_path = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
        test_target_path = os.path.join(folds_dataset_path, test_target_path)

        training_target, validation_target, test_target = uf.get_dataset_matrices(
                    data_path=TARGET_LOAD, num_folds=num_folds, num_fold=num_fold,
                    percs=percs, train_path=train_target_path, val_path=val_target_path,
                    test_path=test_target_path, recompute_matrices=recompute_matrices)
    else:
        #load waveform for both
        training_target = training_predictors
        validation_target = validation_predictors
        test_target = test_predictors


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


    #normalize to 0 mean and unity std (according to training set mean and std)
    '''
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)
    '''

    #sys.exit(0)
    '''
    tr_pred = []
    for i in range(100):
        tr_pred.append(training_predictors[0])
        tr_pred.append(training_predictors[1])

    tr_pred = np.array(tr_pred)
    training_predictors = tr_pred
    validation_predictors = training_predictors
    test_predictors = test_predictors
    '''

    #reshape tensors
    #INSERT HERE FUNCTION FOR CUSTOM RESHAPING!!!!!

    #reshape
    if hybrid_dataset:
        training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
        test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])
    else:
        training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1])
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1])
        test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1])
    training_target = training_target.reshape(training_target.shape[0], 1, training_target.shape[1])
    validation_predictors = validation_target.reshape(validation_target.shape[0], 1, validation_target.shape[1])
    test_predictors = test_target.reshape(test_target.shape[0], 1, test_target.shape[1])

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float().to(device)
    val_predictors = torch.tensor(validation_predictors).float().to(device)
    test_predictors = torch.tensor(test_predictors).float().to(device)

    #build dataset from tensors
    #target i == predictors because autoencoding
    tr_dataset = utils.TensorDataset(train_predictors,train_predictors)
    val_dataset = utils.TensorDataset(val_predictors, val_predictors)
    test_dataset = utils.TensorDataset(test_predictors, test_predictors)

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False)  #no batch here!!
    #DNN input shape
    time_dim = training_predictors.shape[1]
    features_dim = training_predictors.shape[2]
    '''
    if mnist_test:
        #load MNIST
        tr_data = utils.DataLoader(
            datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
        val_data = utils.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
        test_data = val_data
    '''



    #load model (model is in locals()['model'])
    print('\n loading models...')
    encoder_string = 'encoder_class, encoder_parameters = choose_model.' + encoder_architecture + '(time_dim, features_dim, parameters)'
    decoder_string = 'decoder_class, decoder_parameters = choose_model.' + decoder_architecture + '(time_dim, features_dim, parameters)'
    reparametrize_string = 'reparametrize_class = choose_model.' + reparametrize_architecture + '(time_dim, features_dim, parameters)'

    exec(encoder_string)
    exec(decoder_string)
    exec(reparametrize_string)

    encoder = locals()['encoder_class'].to(device)
    decoder = locals()['decoder_class'].to(device)
    reparametrize = locals()['reparametrize_class'].to(device)


    #run training
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    gen_sounds_path = os.path.join(results_path, 'gen_sounds')
    if not os.path.exists(gen_sounds_path):
        os.makedirs(gen_sounds_path)




    #compute number of parameters
    encoder_params = sum([np.prod(p.size()) for p in encoder.parameters()])
    decoder_params = sum([np.prod(p.size()) for p in decoder.parameters()])
    reparametrize_params = sum([np.prod(p.size()) for p in reparametrize.parameters()])

    print ('')
    print ('Encoder paramters: ' + str(encoder_params))
    print ('Decoder paramters: ' + str(decoder_params))
    print ('Reparametrize paramters: ' + str(reparametrize_params))
    print ('Total paramters: ' + str(encoder_params+decoder_params+reparametrize_params))

    #define optimizers
    joint_parameters = list(encoder.parameters()) + list(decoder.parameters())+ list(reparametrize.parameters())
    '''
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate,
                           weight_decay=regularization_lambda)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate,
                           weight_decay=regularization_lambda)
    '''


    optimizer_joint = optim.Adam(joint_parameters, lr=learning_rate,
                           weight_decay=regularization_lambda)


    warm_ramp = losses.warm_up(num_epochs)

    encoder.train()
    decoder.train()
    reparametrize.train()

    total_step = len(tr_data)
    loss_list = []
    train_joint_hist = []
    train_kld_hist = []
    train_recon_hist = []
    val_joint_hist = []
    val_kld_hist = []
    val_recon_hist = []
    patience_vec = []

    #TRAINING LOOP
    #iterate epochs
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        reparametrize.train()

        print ('\n')
        string = 'Epoch: [' + str(epoch+1) + '/' + str(num_epochs) + '] '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)
            #optimizer_encoder.zero_grad()
            #optimizer_decoder.zero_grad()
            optimizer_joint.zero_grad()

            mu, logvar = encoder(sounds)
            z = reparametrize(mu, logvar)
            outputs = decoder(z)
            '''
            sum_mu = torch.sum(torch.sum(mu))
            sum_logvar = torch.sum(torch.sum(logvar))
            print ('culo')
            print (sum_mu, sum_logvar)
            '''

            loss_k = loss_KLD(mu, logvar, epoch, warm_ramp)
            #loss_encoder.backward(retain_graph=True)
            loss_r = loss_recon(outputs, sounds)
            #loss_decoder.backward(retain_graph=True)

            loss_j = loss_joint(outputs, sounds, mu, logvar, epoch, warm_ramp)
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
            #optimizer_encoder.step()
            #optimizer_decoder.step()
            #end of batch loop

        #validation loss, training and val accuracy computation
        #after current epoch training
        encoder.eval()
        decoder.eval()
        reparametrize.eval()

        train_batch_losses_k = []
        val_batch_losses_k = []
        train_batch_losses_r = []
        val_batch_losses_r = []
        train_batch_losses_j = []
        val_batch_losses_j = []

        with torch.no_grad():
            #compute training accuracy and loss
            for i, (sounds, truth) in enumerate(tr_data):
                optimizer_joint.zero_grad()
                sounds = sounds.to(device)
                truth = truth.to(device)

                mu, logvar = encoder(sounds)
                z = reparametrize(mu, logvar)
                outputs = decoder(z)

                loss_k = losses.loss_KLD(mu, logvar, epoch, warm_ramp)
                loss_r = losses.loss_recon(outputs, sounds)
                loss_j = losses.loss_joint(outputs, sounds, mu, logvar, epoch, warm_ramp)

                train_batch_losses_k.append(loss_k.item())
                train_batch_losses_r.append(loss_r.item())
                train_batch_losses_j.append(loss_j.item())

            #compute validation accuracy and loss
            for i, (sounds, truth) in enumerate(val_data):
                optimizer_joint.zero_grad()
                sounds = sounds.to(device)
                truth = truth.to(device)
                mu, logvar = encoder(sounds)
                z = reparametrize(mu, logvar)
                outputs = decoder(z)

                loss_k = losses.loss_KLD(mu, logvar, epoch, warm_ramp)
                loss_r = losses.loss_recon(outputs, sounds)
                loss_j = losses.loss_joint(outputs, sounds, mu, logvar, epoch, warm_ramp)

                val_batch_losses_k.append(loss_k.item())
                val_batch_losses_r.append(loss_r.item())
                val_batch_losses_j.append(loss_j.item())


            train_epoch_kld = np.mean(train_batch_losses_k)
            train_epoch_recon = np.mean(train_batch_losses_r)
            train_epoch_joint = np.mean(train_batch_losses_j)
            val_epoch_kld = np.mean(val_batch_losses_k)
            val_epoch_recon = np.mean(val_batch_losses_r)
            val_epoch_joint = np.mean(val_batch_losses_j)

            train_joint_hist.append(train_epoch_joint)
            train_kld_hist.append(train_epoch_kld)
            train_recon_hist.append(train_epoch_recon)
            val_joint_hist.append(val_epoch_joint)
            val_kld_hist.append(val_epoch_kld)
            val_recon_hist.append(val_epoch_recon)

            print ('\n  train_joint: ' + str(np.round(train_epoch_joint.item(), decimals=5)) + ' | val_joint: ' + str(np.round(val_epoch_joint.item(), decimals=5)))
            print ('  train_KLD: ' + str(np.round(train_epoch_kld.item(), decimals=5)) + ' | val_KLD: ' + str(np.round(val_epoch_kld.item(), decimals=5)))
            print ('  train_recon :' + str(np.round(train_epoch_recon.item(), decimals=5)) + ' | val_recon: ' + str(np.round(val_epoch_recon.item(), decimals=5)))


            #save sounds if specified
            ts_preds = []
            tr_preds = []
            if save_sounds:

                if epoch%save_sounds_epochs == 0: #save only every n epochs

                    curr_sounds_path_train = os.path.join(gen_sounds_path, 'training' , 'epoch_'+str(epoch))
                    curr_sounds_path_test = os.path.join(gen_sounds_path, 'test' , 'epoch_'+str(epoch))
                    curr_orig_path_training = os.path.join(gen_sounds_path, 'training', 'originals')
                    curr_orig_path_test = os.path.join(gen_sounds_path, 'test', 'originals')

                    if not os.path.exists(curr_sounds_path_train):
                        os.makedirs(curr_sounds_path_train)
                    if not os.path.exists(curr_sounds_path_test):
                        os.makedirs(curr_sounds_path_test)
                    if not os.path.exists(curr_orig_path_training):
                        os.makedirs(curr_orig_path_training)
                    if not os.path.exists(curr_orig_path_test):
                        os.makedirs(curr_orig_path_test)

                    #train originals
                    for i, (sounds, truth) in enumerate(tr_data):
                        if i <= save_sounds_n-1:
                            truth = truth.cpu().numpy()
                            truth = truth.flatten()
                            truth = np.divide(truth, np.max(truth))
                            truth = np.multiply(truth, 0.8)
                            sound_name = 'orig_' + str(i) + '.wav'
                            sound_path = os.path.join(curr_orig_path_training, sound_name)
                            uf.wavwrite(truth, SR, sound_path)
                        else:
                            break

                    #test originals
                    for i, (sounds, truth) in enumerate(test_data):
                        if i <= save_sounds_n-1:
                            truth = truth.cpu().numpy()
                            truth = truth.flatten()
                            truth = np.divide(truth, np.max(truth))
                            truth = np.multiply(truth, 0.8)
                            sound_name = 'orig_' + str(i) + '.wav'
                            sound_path = os.path.join(curr_orig_path_test, sound_name)
                            uf.wavwrite(truth, SR, sound_path)
                        else:
                            break

                    #train sounds
                    for i, (sounds, truth) in enumerate(tr_data):  #save n predictions from test set
                        #optimizer_encoder.zero_grad()
                        optimizer_joint.zero_grad()
                        sounds = sounds.to(device)
                        truth = truth.to(device)

                        mu, logvar = encoder(sounds)
                        z = reparametrize(mu, logvar)
                        outputs = decoder(z)


                        outputs = outputs.cpu().numpy()
                        for single_sound in outputs:
                            tr_preds.append(single_sound)

                    tr_preds = np.array(tr_preds)


                    #tr_preds = tr_preds.reshape(tr_preds.shape[0]*tr_preds.shape[1], tr_preds.shape[2], tr_preds.shape[3])
                    for i in range(save_sounds_n):
                        sound = tr_preds[i]
                        sound = sound.flatten()
                        #print (sound[:10])
                        #normalize
                        sound = np.divide(sound, np.max(sound))
                        sound = np.multiply(sound, 0.8)
                        sound_name = 'gen_' + str(i) + '.wav'
                        sound_path = os.path.join(curr_sounds_path_train, sound_name)
                        uf.wavwrite(sound, SR, sound_path)

                    #test sounds
                    for i, (sounds, truth) in enumerate(test_data):  #save n predictions from test set
                        #optimizer_encoder.zero_grad()
                        optimizer_joint.zero_grad()
                        sounds = sounds.to(device)
                        truth = truth.to(device)

                        mu, logvar = encoder(sounds)
                        z = reparametrize(mu, logvar)
                        outputs = decoder(z)

                        outputs = outputs.cpu().numpy()
                        for single_sound in outputs:
                            ts_preds.append(single_sound)

                    ts_preds = np.array(ts_preds)
                    #ts_preds = ts_preds.reshape(ts_preds.shape[0]*ts_preds.shape[1], ts_preds.shape[2], ts_preds.shape[3])
                    for i in range(save_sounds_n):
                        sound = ts_preds[i]
                        sound = sound.flatten()
                        #normalize
                        sound = np.divide(sound, np.max(sound))
                        sound = np.multiply(sound, 0.8)
                        sound_name = 'gen_' + str(i) + '.wav'
                        sound_path = os.path.join(curr_sounds_path_test, sound_name)
                        uf.wavwrite(sound, SR, sound_path)


                    print ('')
                    print ('generated sounds saved')

            #end of epoch loop
        '''
        #compute train and val mean accuracy and loss of current epoch
        train_epoch_loss_e = np.mean(train_batch_losses_e)
        train_epoch_loss_d = np.mean(train_batch_losses_d)
        val_epoch_loss_e = np.mean(val_batch_losses_e)
        val_epoch_loss_d = np.mean(val_batch_losses_d)

        #append values to histories
        train_loss_hist_e.append(train_epoch_loss_e)
        train_loss_hist_d.append(train_epoch_loss_d)
        val_loss_hist_e.append(val_epoch_loss_e)
        val_loss_hist_d.append(val_epoch_loss_d)

        #print loss and accuracy of the current epoch
        print ('\r', 'train_loss_encoder: ' + str(train_epoch_loss) + '| val_loss: ' + str(val_epoch_loss))

        #save best model (metrics = loss)
        if save_best_only == True:
            if epoch == 0:
                torch.save(model.state_dict(), BVL_model_path)
                print ('saved_BVL')
                saved_epoch = epoch + 1
            else:
                best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]
                if curr_loss < best_loss:
                    torch.save(model.state_dict(), BVL_model_path)
                    print ('saved_BVL')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1

        utilstring = 'dataset: ' + str(dataset) + ', exp: ' + str(num_experiment) + ', run: ' + str(num_run) + ', fold: ' + str(num_fold)
        print (utilstring)


        #early stopping
        if early_stopping and epoch >= patience:
            prev_loss = val_hist[-2]
            curr_loss = val_hist[-1]
            if curr_loss < prev_loss:
                patience_vec = []
            else:
                patience_vec.append(curr_loss)
                if len(patience_vec) == patience:
                    print ('\n')
                    print ('Training stopped with patience = ' + str(patience) + ', saved at epoch = ' + str(saved_epoch))
                    break

        #AS LAST THING, AFTER OPTIMIZER.STEP AND EVENTUAL MODEL SAVING
        #AVERAGE MULTISCALE CONV KERNELS!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            model.multiscale1.update_kernels()
            model.multiscale2.update_kernels()
            model.multiscale3.update_kernels()
        except:
            pass

        #END OF EPOCH

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
