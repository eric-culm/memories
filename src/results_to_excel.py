from __future__ import print_function
import numpy as np
import sys, os
import xlsxwriter

try:
    in_folder = sys.argv[1]
    out_name = sys.argv[2]
except:
    pass

#read results folder
contents = os.listdir(in_folder)
num_exps = len(contents)
out_name = os.path.join(in_folder, out_name)
contents = list(filter(lambda x: '.npy' in x, contents))
temp_path = os.path.join(in_folder, contents[0])
dict = np.load(temp_path)
dict = dict.item()

#find if is regression or classification task
try:
    dummy = dict['summary']['training']['mean_MAE']
    task_type = 'regression'
except:
    task_type = 'classification'


#init workbook
workbook = xlsxwriter.Workbook(out_name)
worksheet = workbook.add_worksheet()

#define styles
values_format = workbook.add_format({'align': 'center','border': 1})
blank_format = workbook.add_format({'align': 'center','border': 1})
bestvalue_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'green'})
bestvalueSTD_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'blue'})
header_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'green'})
parameters_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'yellow'})

loss_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'red'})

accuracy_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'orange'})
f1_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'cyan'})
precision_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'lime'})
recall_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'magenta'})

rmse_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'orange'})
mae_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'cyan'})
percs_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'#800080'})
separation_border = workbook.add_format({'border': 1,'bottom': 6, 'bottom_color':'#ff0000'})

#define column names
exp_id_c = 0
parameters_c = 1
comment1_c = 2
comment2_c = 3

train_loss_c = 4
val_loss_c = 5
test_loss_c = 6
train_loss_std_c = 7
val_loss_std_c = 8
test_loss_std_c = 9

if task_type == 'regression':
    train_rmse_c = 10
    val_rmse_c = 11
    test_rmse_c = 12
    train_rmse_std_c = 13
    val_rmse_std_c = 14
    test_rmse_std_c = 15

    train_mae_c = 16
    val_mae_c = 17
    test_mae_c = 18
    train_mae_std_c = 19
    val_mae_std_c = 20
    test_mae_std_c = 21

    end_c = 21

elif task_type == 'classification':
    train_acc_c = 10
    val_acc_c = 11
    test_acc_c = 12
    train_acc_std_c = 13
    val_acc_std_c = 14
    test_acc_std_c = 15

    train_f1_c = 16
    val_f1_c = 17
    test_f1_c = 18
    train_f1_std_c = 19
    val_f1_std_c = 20
    test_f1_std_c = 21

    train_precision_c = 22
    val_precision_c = 23
    test_precision_c = 24
    train_precision_std_c = 25
    val_precision_std_c = 26
    test_precision_std_c = 27

    train_recall_c = 28
    val_recall_c = 29
    test_recall_c = 30
    train_recall_std_c = 31
    val_recall_std_c = 32
    test_recall_std_c = 33

    end_c = 33


v_offset = 2
v_end = v_offset + num_exps + 1


#write header
#title
worksheet.merge_range(v_offset-2, exp_id_c, v_offset-2, end_c, "RESULTS", header_format)
#parameters
worksheet.merge_range(v_offset-1,exp_id_c, v_offset-1, comment2_c, "PARAMETERS", parameters_format)
#mean and std acc, loss, f1, precision, recall
#loss
worksheet.merge_range(v_offset-1,train_loss_c, v_offset-1, test_loss_c, "MEAN LOSS", loss_format)
worksheet.merge_range(v_offset-1,train_loss_std_c, v_offset-1, test_loss_std_c, "LOSS STD", loss_format)
if task_type == 'regression':
    #rmse
    worksheet.merge_range(v_offset-1,train_rmse_c, v_offset-1, test_rmse_c, "MEAN RMSE", rmse_format)
    worksheet.merge_range(v_offset-1,train_rmse_std_c, v_offset-1, test_rmse_std_c, "RMSE STD", rmse_format)
    #mae
    worksheet.merge_range(v_offset-1,train_mae_c, v_offset-1, test_mae_c, "MEAN MAE", mae_format)
    worksheet.merge_range(v_offset-1,train_mae_std_c, v_offset-1, test_mae_std_c, "MAE STD", mae_format)
elif task_type == 'classification':
    #acc
    worksheet.merge_range(v_offset-1,train_acc_c, v_offset-1, test_acc_c, " MEAN ACCURACY", accuracy_format)
    worksheet.merge_range(v_offset-1,train_acc_std_c, v_offset-1, test_acc_std_c, "ACCURACY STD", accuracy_format)
    #f1
    worksheet.merge_range(v_offset-1,train_f1_c, v_offset-1, test_f1_c, " MEAN F1", f1_format)
    worksheet.merge_range(v_offset-1,train_f1_std_c, v_offset-1, test_f1_std_c, "F1 STD", f1_format)
    #precision
    worksheet.merge_range(v_offset-1,train_precision_c, v_offset-1, test_precision_c, " MEAN PRECISION", precision_format)
    worksheet.merge_range(v_offset-1,train_precision_std_c, v_offset-1, test_precision_std_c, "PRECISION STD", precision_format)
    #recall
    worksheet.merge_range(v_offset-1,train_recall_c, v_offset-1, test_recall_c, " MEAN RECALL", recall_format)
    worksheet.merge_range(v_offset-1,train_recall_std_c, v_offset-1, test_recall_std_c, "RECALL STD", recall_format)

#write column names
worksheet.write(v_offset, exp_id_c, 'ID',parameters_format)
worksheet.set_column(exp_id_c,exp_id_c,6)

worksheet.write(v_offset, comment1_c, 'comment 1',parameters_format)
worksheet.set_column(comment1_c,comment1_c,35)

worksheet.write(v_offset, comment2_c, 'comment 2',parameters_format)
worksheet.set_column(comment2_c,comment2_c,35)

worksheet.write(v_offset, parameters_c, 'link to parameters',parameters_format)
worksheet.set_column(parameters_c,parameters_c,35)

worksheet.write(v_offset, train_loss_c, 'train',loss_format)
worksheet.write(v_offset, val_loss_c, 'val',loss_format)
worksheet.write(v_offset, test_loss_c, 'test',loss_format)
worksheet.write(v_offset, train_loss_std_c, 'train',loss_format)
worksheet.write(v_offset, val_loss_std_c, 'val',loss_format)
worksheet.write(v_offset, test_loss_std_c, 'test',loss_format)

if task_type == 'regression':
        worksheet.write(v_offset, train_rmse_c, 'train',rmse_format)
        worksheet.write(v_offset, val_rmse_c, 'val',rmse_format)
        worksheet.write(v_offset, test_rmse_c, 'test',rmse_format)

        worksheet.write(v_offset, train_mae_c, 'train',mae_format)
        worksheet.write(v_offset, val_mae_c, 'val',mae_format)
        worksheet.write(v_offset, test_mae_c, 'test',mae_format)

        worksheet.write(v_offset, train_rmse_std_c, 'train',rmse_format)
        worksheet.write(v_offset, val_rmse_std_c, 'val',rmse_format)
        worksheet.write(v_offset, test_rmse_std_c, 'test',rmse_format)

        worksheet.write(v_offset, train_mae_std_c, 'train',mae_format)
        worksheet.write(v_offset, val_mae_std_c, 'val',mae_format)
        worksheet.write(v_offset, test_mae_std_c, 'test',mae_format)
elif task_type == 'classification':
    worksheet.write(v_offset, train_acc_c, 'train',accuracy_format)
    worksheet.write(v_offset, val_acc_c, 'val',accuracy_format)
    worksheet.write(v_offset, test_acc_c, 'test',accuracy_format)

    worksheet.write(v_offset, train_f1_c, 'train',f1_format)
    worksheet.write(v_offset, val_f1_c, 'val',f1_format)
    worksheet.write(v_offset, test_f1_c, 'test',f1_format)

    worksheet.write(v_offset, train_precision_c, 'train',precision_format)
    worksheet.write(v_offset, val_precision_c, 'val',precision_format)
    worksheet.write(v_offset, test_precision_c, 'test',precision_format)

    worksheet.write(v_offset, train_recall_c, 'train',recall_format)
    worksheet.write(v_offset, val_recall_c, 'val',recall_format)
    worksheet.write(v_offset, test_recall_c, 'test',recall_format)

    worksheet.write(v_offset, train_acc_std_c, 'train',accuracy_format)
    worksheet.write(v_offset, val_acc_std_c, 'val',accuracy_format)
    worksheet.write(v_offset, test_acc_std_c, 'test',accuracy_format)

    worksheet.write(v_offset, train_f1_std_c, 'train',f1_format)
    worksheet.write(v_offset, val_f1_std_c, 'val',f1_format)
    worksheet.write(v_offset, test_f1_std_c, 'test',f1_format)

    worksheet.write(v_offset, train_precision_std_c, 'train',precision_format)
    worksheet.write(v_offset, val_precision_std_c, 'val',precision_format)
    worksheet.write(v_offset, test_precision_std_c, 'test',precision_format)

    worksheet.write(v_offset, train_recall_std_c, 'train',recall_format)
    worksheet.write(v_offset, val_recall_std_c, 'val',recall_format)
    worksheet.write(v_offset, test_recall_std_c, 'test',recall_format)

    worksheet.set_column(train_acc_c,test_loss_std_c,10)

#fill values
#iterate every experiment
for i in contents:
    if '.npy' in i:
        temp_path = os.path.join(in_folder, i)
        dict = np.load(temp_path)
        dict = dict.item()
        keys = dict[0].keys()

        #print grobal parameters
        #write ID
        exp_ID = i.split('_')[-1].split('.')[0][3:]
        curr_row = int(exp_ID)+v_offset
        worksheet.write(curr_row, exp_id_c, exp_ID,values_format)
        #write comment
        parameters = dict['summary']['parameters'].split('/')
        comment_1 = '/'
        comment_2 = '/'
        for par in parameters:
            if 'comment_1' in par:
                comment_1 = par.split('=')[1].replace('"', '')
            if 'comment_2' in par:
                comment_2 = par.split('=')[1].replace('"', '')
        worksheet.write(curr_row, comment1_c, comment_1,values_format)
        worksheet.write(curr_row, comment2_c, comment_2,values_format)

        #write parameters link
        split_par = i.split('.')[0].split('_')
        curr_par_name = 'parameters/parameters_' + split_par[1] + '_' + split_par[2] + '_' + split_par[3] + '.txt'
        worksheet.write_url(curr_row, parameters_c, 'external:'+curr_par_name)

        #extract losses
        tr = dict['summary']['training']
        val = dict['summary']['validation']
        test = dict['summary']['test']

        tr_loss = tr['mean_loss']
        tr_loss_std = tr['loss_std']
        val_loss = val['mean_loss']
        val_loss_std = val['loss_std']
        test_loss = test['mean_loss']
        test_loss_std = test['loss_std']

        if task_type == 'regression':
            #training results
            tr_rmse = tr['mean_RMSE']
            tr_mae = tr['mean_MAE']
            tr_rmse_std = tr['RMSE_std']
            tr_mae_std = tr['MAE_std']
            #val results
            val_rmse = val['mean_RMSE']
            val_mae = val['mean_MAE']
            val_rmse_std = val['RMSE_std']
            val_mae_std = val['MAE_std']
            #test results
            test_rmse = test['mean_RMSE']
            test_mae = test['mean_MAE']
            test_rmse_std = test['RMSE_std']
            test_mae_std = test['MAE_std']
        elif task_type == 'classification':
            #extract results
            #training results
            tr_acc = tr['mean_acc']
            tr_f1 = tr['mean_f1']
            tr_precision = tr['mean_precision']
            tr_recall = tr['mean_recall']
            tr_acc_std = tr['acc_std']
            tr_f1_std = tr['f1_std']
            tr_precision_std = tr['precision_std']
            tr_recall_std = tr['recall_std']
            #validation results
            val_acc = val['mean_acc']
            val_f1 = val['mean_f1']
            val_precision = val['mean_precision']
            val_recall = val['mean_recall']
            val_acc_std = val['acc_std']
            val_f1_std = val['f1_std']
            val_precision_std = val['precision_std']
            val_recall_std = val['recall_std']
            #test results
            test_acc = test['mean_acc']
            test_f1 = test['mean_f1']
            test_precision = test['mean_precision']
            test_recall = test['mean_recall']
            test_acc_std = test['acc_std']
            test_f1_std = test['f1_std']
            test_precision_std = test['precision_std']
            test_recall_std = test['recall_std']

        #print results
        #loss
        worksheet.write(curr_row, train_loss_c, tr_loss,values_format)
        worksheet.write(curr_row, val_loss_c, val_loss,values_format)
        worksheet.write(curr_row, test_loss_c, test_loss,values_format)
        #loss std
        worksheet.write(curr_row, train_loss_std_c, tr_loss_std,values_format)
        worksheet.write(curr_row, val_loss_std_c, val_loss_std,values_format)
        worksheet.write(curr_row, test_loss_std_c, test_loss_std,values_format)
        if task_type == 'regression':
            #rmse
            worksheet.write(curr_row, train_rmse_c, tr_rmse,values_format)
            worksheet.write(curr_row, val_rmse_c, val_rmse,values_format)
            worksheet.write(curr_row, test_rmse_c, test_rmse,values_format)
            #rmse std
            worksheet.write(curr_row, train_rmse_std_c, tr_rmse_std,values_format)
            worksheet.write(curr_row, val_rmse_std_c, val_rmse_std,values_format)
            worksheet.write(curr_row, test_rmse_std_c, test_rmse_std,values_format)
            #mae
            worksheet.write(curr_row, train_mae_c, tr_mae,values_format)
            worksheet.write(curr_row, val_mae_c, val_mae,values_format)
            worksheet.write(curr_row, test_mae_c, test_mae,values_format)
            #mae std
            worksheet.write(curr_row, train_mae_std_c, tr_mae_std,values_format)
            worksheet.write(curr_row, val_mae_std_c, val_mae_std,values_format)
            worksheet.write(curr_row, test_mae_std_c, test_mae_std,values_format)
        elif task_type == 'classification':
            #acc
            worksheet.write(curr_row, train_acc_c, tr_acc,values_format)
            worksheet.write(curr_row, val_acc_c, val_acc,values_format)
            worksheet.write(curr_row, test_acc_c, test_acc,values_format)
            #acc std
            worksheet.write(curr_row, train_acc_std_c, tr_acc_std,values_format)
            worksheet.write(curr_row, val_acc_std_c, val_acc_std,values_format)
            worksheet.write(curr_row, test_acc_std_c, test_acc_std,values_format)
            #f1
            worksheet.write(curr_row, train_f1_c, tr_f1,values_format)
            worksheet.write(curr_row, val_f1_c, val_f1,values_format)
            worksheet.write(curr_row, test_f1_c, test_f1,values_format)
            #f1 std
            worksheet.write(curr_row, train_f1_std_c, tr_f1_std,values_format)
            worksheet.write(curr_row, val_f1_std_c, val_f1_std,values_format)
            worksheet.write(curr_row, test_f1_std_c, test_f1_std,values_format)
            #precision
            worksheet.write(curr_row, train_precision_c, tr_precision,values_format)
            worksheet.write(curr_row, val_precision_c, val_precision,values_format)
            worksheet.write(curr_row, test_precision_c, test_precision,values_format)
            #precision std
            worksheet.write(curr_row, train_precision_std_c, tr_precision_std,values_format)
            worksheet.write(curr_row, val_precision_std_c, val_precision_std,values_format)
            worksheet.write(curr_row, test_precision_std_c, test_precision_std,values_format)
            #recall
            worksheet.write(curr_row, train_recall_c, tr_recall,values_format)
            worksheet.write(curr_row, val_recall_c, val_recall,values_format)
            worksheet.write(curr_row, test_recall_c, test_recall,values_format)
            #recall std
            worksheet.write(curr_row, train_recall_std_c, tr_recall_std,values_format)
            worksheet.write(curr_row, val_recall_std_c, val_recall_std,values_format)
            worksheet.write(curr_row, test_recall_std_c, test_recall_std,values_format)


explist = list(range(1, num_exps+1))
endlist = []
#apply blank formatting to blank and non-ending lines
#this is necessary due to a bug
for end in explist:
    #if end not in endlist:
    if end not in []:
        worksheet.conditional_format( v_offset+end,0,v_offset+end,end_c, {'type': 'blanks','format': blank_format})

#highlight best values
#loss
worksheet.conditional_format(v_offset, train_loss_c, v_offset+num_exps, train_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, val_loss_c, v_offset+num_exps, val_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, test_loss_c, v_offset+num_exps, test_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})
#loss std
worksheet.conditional_format(v_offset, train_loss_std_c, v_offset+num_exps, train_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, val_loss_std_c, v_offset+num_exps, val_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, test_loss_std_c, v_offset+num_exps, test_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})
if task_type == 'regression':
    #rmse
    worksheet.conditional_format(v_offset, train_rmse_c, v_offset+num_exps, train_rmse_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_rmse_c, v_offset+num_exps, val_rmse_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_rmse_c, v_offset+num_exps, test_rmse_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})
    #rmse std
    worksheet.conditional_format(v_offset, train_rmse_std_c, v_offset+num_exps, train_rmse_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_rmse_std_c, v_offset+num_exps, val_rmse_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_rmse_std_c, v_offset+num_exps, test_rmse_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})
    #mae
    worksheet.conditional_format(v_offset, train_mae_c, v_offset+num_exps, train_mae_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_mae_c, v_offset+num_exps, val_mae_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_mae_c, v_offset+num_exps, test_mae_c,
                                    {'type': 'bottom','value': '1','format': bestvalue_format})
    #mae std
    worksheet.conditional_format(v_offset, train_mae_std_c, v_offset+num_exps, train_mae_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_mae_std_c, v_offset+num_exps, val_mae_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_mae_std_c, v_offset+num_exps, test_mae_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})
elif task_type == 'classification':
#acc
    worksheet.conditional_format(v_offset, train_acc_c, v_offset+num_exps, train_acc_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_acc_c, v_offset+num_exps, val_acc_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_acc_c, v_offset+num_exps, test_acc_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})
    #acc std
    worksheet.conditional_format(v_offset, train_acc_std_c, v_offset+num_exps, train_acc_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_acc_std_c, v_offset+num_exps, val_acc_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_acc_std_c, v_offset+num_exps, test_acc_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})
    #f1
    worksheet.conditional_format(v_offset, train_f1_c, v_offset+num_exps, train_f1_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_f1_c, v_offset+num_exps, val_f1_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_f1_c, v_offset+num_exps, test_f1_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})
    #f1 std
    worksheet.conditional_format(v_offset, train_f1_std_c, v_offset+num_exps, train_f1_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_f1_std_c, v_offset+num_exps, val_f1_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_f1_std_c, v_offset+num_exps, test_f1_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})
    #precision
    worksheet.conditional_format(v_offset, train_precision_c, v_offset+num_exps, train_precision_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_precision_c, v_offset+num_exps, val_precision_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_precision_c, v_offset+num_exps, test_precision_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})
    #precision std
    worksheet.conditional_format(v_offset, train_precision_std_c, v_offset+num_exps, train_precision_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_precision_std_c, v_offset+num_exps, val_precision_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_precision_std_c, v_offset+num_exps, test_precision_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})
    #recall
    worksheet.conditional_format(v_offset, train_recall_c, v_offset+num_exps, train_recall_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, val_recall_c, v_offset+num_exps, val_recall_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset, test_recall_c, v_offset+num_exps, test_recall_c,
                                    {'type': 'top','value': '1','format': bestvalue_format})
    #recall std
    worksheet.conditional_format(v_offset, train_recall_std_c, v_offset+num_exps, train_recall_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, val_recall_std_c, v_offset+num_exps, val_recall_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

    worksheet.conditional_format(v_offset, test_recall_std_c, v_offset+num_exps, test_recall_std_c,
                                    {'type': 'bottom','value': '1','format': bestvalueSTD_format})

workbook.close()
