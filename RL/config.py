
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=2, help='actor LSTM num_neurons')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default= 128, help='batch size')
data_arg.add_argument('--input_dimension', type=int, default=4, help='city dimension')
data_arg.add_argument('--nCells', type=int, default=10, help='number of cells')
data_arg.add_argument('--nMuts', type=int, default=7, help='number of mutations')
data_arg.add_argument('--input_dir_n', type=str, help='noisy matrices directory')
data_arg.add_argument('--input_dir_p', type=str,  help='perfect matrices directory')
data_arg.add_argument('--output_dir', type=str,  help='output matrices directory')
data_arg.add_argument('--table_dir', type=str,  help='output table directory')
data_arg.add_argument('--nTrain', type=int, default=10000, help='number of train instances')
data_arg.add_argument('--nTestMats', type=int, default=100, help='number of test instances')
data_arg.add_argument('--fp', type=float, default=0.05, help='False positive rate')
data_arg.add_argument('--fn', type=float, default=0.0001, help='False negative rate')
data_arg.add_argument('--betta', type=float, default=0.04, help='hyperparameter for adjusting NLL')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=1500, help='nb epoch')

# Misc
misc_arg = add_argument_group('User options') 

misc_arg.add_argument('--inference_mode', type=str2bool, default=True, help='switch to inference mode when model is trained') 
misc_arg.add_argument('--restore_model', type=str2bool, default=True, help='whether or not model is retrieved')
misc_arg.add_argument('--active_search', type=str2bool, default=True, help='active search using a pre-trained model')

misc_arg.add_argument('--save_to', type=str, default='model', help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='model', help='loader sub directory')  ###
misc_arg.add_argument('--ms_dir', type=str, default='/u/ms', help='ms program directory') 



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_config():
    config, _ = get_config()
    print('\n')
    print('Data Config:')
    print('* Batch size:',config.batch_size)
    print('* Sequence length:',config.nCells*config.nMuts)
    # print('* Starting batch number:', config.starting_num)
    print('\n')
    print('Network Config:')
    print('* Restored model:',config.restore_model)
    print('* Actor hidden_dim:',config.hidden_dim*config.input_dimension)
    # print('* Actor tan clipping:',config.C)
    print('\n')
    if config.inference_mode==False:
        print('Training Config:')
        print('* Nb epoch:',config.nb_epoch)
        # print('* Temperature:',config.temperature)
        # print('* Actor learning rate (init,decay_step,decay_rate):',config.lr1_start,config.lr1_decay_step,config.lr1_decay_rate)
    else:
        print('Testing Config:')
        print('Number of test matrices:', config.nTestMats)
    # print('* Summary writer log dir:',config.log_dir)
    print('\n')