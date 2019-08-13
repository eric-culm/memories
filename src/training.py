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
    dataset = 'sc09'
    architecture = 'dummy_autoencoder'
    parameters = ['niente = 0']
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
    print ('saving model at: ' + SAVE_MODEL)
    print ('')


import loadconfig
import configparser
import json
import torch
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
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)

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

#path for saving best val loss and best val acc models
BVL_model_path = SAVE_MODEL

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
    for param in parameters:
        exec(param)

except IndexError:
    pass

device = torch.device('cuda:' + str(gpu_ID))

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
    predictors_merged = np.load(PREDICTORS_LOAD)
    predictors_merged = predictors_merged.item()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
    #TRAIN/VAL/TEST IN A BALANCED WAY
    foldable_list = list(predictors_merged.keys())
    fold_actors_list = uf.folds_generator(num_folds, foldable_list, percs)
    train_list = fold_actors_list[int(num_fold)]['train']
    val_list = fold_actors_list[int(num_fold)]['val']
    test_list = fold_actors_list[int(num_fold)]['test']
    #del dummy

    sys.exit(0)

    #if tensors of current fold has not been computed:
    if not os.path.exists(test_target_path):
        #load merged dataset, compute and save current tensors

        #predictors_merged = np.load(PREDICTORS_LOAD)
        #predictors_merged = predictors_merged.item()

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
    #reshape
    training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
    validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
    test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

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


    #load and compile model (model is in locals()['model'])
    print('\n loading model...')
    model_string = 'model, model_parameters = choose_model.' + architecture + '(time_dim, features_dim, parameters)'
    exec(model_string)
    locals()['model'].compile(loss=loss_function, optimizer=opt, metrics=metrics_list)
    print (locals()['model'].summary())
    #print (locals()['model_parameters'])


    #run training
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


    #compute number of parameters
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('')
    print ('Total paramters: ' + str(tot_params))

    #define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=regularization_lambda)
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_step = len(tr_data)
    loss_list = []
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    patience_vec = []

    #TRAINING LOOP
    #iterate epochs
    for epoch in range(num_epochs):
        model.train()
        print ('\n')
        string = 'Epoch: ' + str(epoch+1) + ' '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            outputs = model(sounds)
            loss = criterion(outputs, truth)
            loss.backward()
            #print progress and update history, optimizer step
            perc = int(i / len(tr_data) * 20)
            inv_perc = int(20 - perc - 1)
            loss_print_t = str(np.round(loss.item(), decimals=3))
            string2 = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
            print ('\r', string2, end='')
            optimizer.step()
            #end of batch loop

        #validation loss, training and val accuracy computation
        #after current epoch training
        model.eval()
        train_batch_losses = []
        val_batch_losses = []
        with torch.no_grad():
            #compute training accuracy and loss
            for i, (sounds, truth) in enumerate(tr_data):
                optimizer.zero_grad()
                tr_outputs = model(sounds)
                temp_tr_loss = criterion(tr_outputs, truth)
                train_batch_losses.append(temp_tr_loss.item())
            #compute validation accuracy and loss
            for i, (sounds, truth) in enumerate(val_data):
                optimizer.zero_grad()
                val_outputs = model(sounds)
                temp_val_loss = criterion(val_outputs, truth)
                val_batch_losses.append(temp_val_loss.item())
            #end of epoch loop

        #compute train and val mean accuracy and loss of current epoch
        train_epoch_loss = np.mean(train_batch_losses)
        train_epoch_acc = np.mean(train_batch_accs)
        val_epoch_loss = np.mean(val_batch_losses)
        val_epoch_acc = np.mean(val_batch_accs)

        #append values to histories
        train_loss_hist.append(train_epoch_loss)
        train_acc_hist.append(train_epoch_acc)
        val_loss_hist.append(val_epoch_loss)
        val_acc_hist.append(val_epoch_acc)


        #print loss and accuracy of the current epoch
        print ('\r', 'train_loss: ' + str(train_epoch_loss) + '| val_loss: ' + str(val_epoch_loss))

        #save best model (metrics = loss)
        if save_best_only == True:
            if epoch == 0:
                torch.save(model.state_dict(), BVL_model_path)
                print ('saved_BVL')
                saved_epoch = epoch + 1
            else:
                best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
                best_acc = max(val_acc_hist[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]
                curr_acc = val_acc_hist[-1]
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
        if training_mode == 'train_and_eval' or training_mode == 'only_gradient' or training_mode == 'only_train':
            model.multiscale1.update_kernels()
            if network_type == '3_layer':
                model.multiscale2.update_kernels()
                model.multiscale3.update_kernels()
        elif training_mode =='only_eval':
            pass
        else:
            raise NameError ('Invalid training mode')
            print ('Given mode: ' + str(training_mode))

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
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
            train_batch_losses.append(temp_loss)
        #val acc
        for i, (sounds, truth) in enumerate(val_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
            val_batch_losses.append(temp_loss)
        #test acc
        for i, (sounds, truth) in enumerate(test_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
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
    #save accuracy
    temp_results['train_acc_BVL'] = train_acc_BVL
    temp_results['val_acc_BVL'] = val_acc_BVL
    temp_results['test_acc_BVL'] = test_acc_BVL
    temp_results['train_acc_BVA'] = train_acc_BVA
    temp_results['val_acc_BVA'] = val_acc_BVA
    temp_results['test_acc_BVA'] = test_acc_BVA




    #save results in temp dict file
    temp_results = {}



    np.save(results_path, temp_results)


if __name__ == '__main__':
    main()















#
