#/usr/bin/python3


# import tensorflow as tf
# from tensorflow import keras
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Embedding, BatchNormalization, Reshape, Conv1D, Activation, add, Lambda, Layer, LSTM, LSTMCell, Add, Concatenate, multiply, Dense, Subtract
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.initializers import glorot_uniform
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from keras.utils import CustomObjectScope


from config import get_config, print_config
from dataset import DataSet
from attention import AttentionLayer
from tensorflow.random import categorical
from tensorflow.contrib.distributions import Categorical


from encoder import encoder
from decoder import decoder
from utils import *
from critic import critic
from cost import count3gametes, cost
from predict import solve


if __name__ == "__main__":

    ########################### configs ###########################
    config, _ = get_config()
    K.set_learning_phase(1)
    K.tensorflow_backend._get_available_gpus()
    K.clear_session()

    if not config.inference_mode:
    ########################### Inputs ###########################
        input_data = Input(shape = (config.nCells * config.nMuts, config.input_dimension), name = 'main_input')
     

        ########################### Encoder ###########################
        encoded_input = encoder(input_data, config)
        n_hidden = K.int_shape(encoded_input)[2] # num_neurons

        ########################### Critic ############################
        critic_predictions = critic(input_data, n_hidden, config)


        ########################### Decoder ###########################
        f = Input(batch_shape = (config.batch_size, n_hidden), name = 'f_input')  # random tensor as first input of decoder
        poss, log_s = decoder(encoded_input, f, n_hidden, config)

        cost_layer = Cost(poss, config, name = 'Cost_layer')
        cost_v = cost_layer(input_data)

        reward_baseline_layer = StopGrad(name = 'stop_gradient') #critic_predictions,
        reward_baseline = reward_baseline_layer([critic_predictions, cost_v])

        AdamOpt_actor = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)  
        AdamOpt_critic = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)

        model_critic = Model(inputs = input_data, outputs = critic_predictions)  
        model_critic.compile(loss = costum_loss1(critic_predictions, cost_v), optimizer = AdamOpt_critic)


        model_actor = Model(inputs = [input_data, f], outputs = poss)  
        model_actor.compile(loss = costum_loss(reward_baseline, log_s), optimizer = AdamOpt_actor)

        print_config()

    
        dataset = DataSet(config)

        fn = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fn_r.txt')]
        fp = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fp_r.txt')]
        l = []
        for i in range(config.nCells):
            for j in range(config.nMuts):
                l.append([i,j])
        l = np.asarray(l)

        f_input = np.random.randn(config.batch_size, n_hidden)
        target_actor = [np.zeros((config.batch_size))] 
        target_critic = [np.zeros((config.batch_size))]
        act_loss = []
        crc_loss = []

        st = time()
        for i in tqdm(range(config.starting_num, config.nb_epoch)): # epoch i


            actor_loss = model_actor.train_on_batch([dataset.train_batch(i, fp, fn, l), f_input], target_actor)
            critic_loss = model_critic.train_on_batch(dataset.train_batch(i, fp, fn, l), target_critic)
            act_loss.append(actor_loss)
            crc_loss.append(critic_loss)

            if (i % 1000 == 0):
                # serialize model to JSON
                model_json = model_actor.to_json()
                with open(config.save_to + "/model_{}.json".format(i), "w") as json_file:
                    json_file.write(model_json)

                model_actor.save_weights(config.save_to + "/model_{}.h5".format(i))
                print("Saved model to disk")
            if i == config.nb_epoch - 1:

                model_json = model_actor.to_json()
                with open(config.save_to + "/actor.json", "w") as json_file:
                    json_file.write(model_json)

                model_actor.save_weights(config.save_to + "/actor.h5")
                print("Saved actor to disk")

        print("time:", time() - st)
        f, axes = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace = 0.35)


        axes[0].plot(act_loss)
        axes[0].set_title('training loss, actor')
        axes[0].set(xlabel = 'batch' , ylabel = 'loss')
        axes[0].legend(['actor training'], loc='upper left')

        axes[1].plot(crc_loss)
        axes[1].set_title('training loss, critic')
        axes[1].set(xlabel = 'batch' , ylabel = 'loss')
        axes[1].legend(['critic training'], loc='upper left')



        plt.savefig('{}.png'.format('loss_{nCells}x{nMuts}'.format(nCells = config.nCells, nMuts = config.nMuts)))
        plt.close(f)



    else:

        with CustomObjectScope({'Simple': Simple, 'Decode': Decode, 'StopGrad' : StopGrad, 'Cost' : Cost, 'AttentionLayer': AttentionLayer}):
            json_file = open(config.restore_from + '/actor.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            # load weights into new model
            loaded_model.load_weights(config.restore_from + "/actor.h5")
            print("Loaded model from disk")

        n_hidden = 8
        solve(loaded_model, config, n_hidden)






