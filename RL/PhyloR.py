
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam

from data import *
from config import get_config, print_config
from predict import solve
from utils import *

from encoder import encoder
from decoder import decoder
from utils import *
from critic import critic



if __name__ == '__main__':
    ########################### Configs ###########################
    config, _ = get_config()
    K.set_learning_phase(1)
    K.tensorflow_backend._get_available_gpus()
    K.clear_session()

    print_config()
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

    reward_baseline_layer = StopGrad(name = 'stop_gradient') 
    reward_baseline = reward_baseline_layer([critic_predictions, cost_v])

    ######################### Models ##############################
    AdamOpt_actor = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)  
    AdamOpt_critic = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, amsgrad=False)

    model_critic = Model(inputs = input_data, outputs = critic_predictions)  
    model_critic.compile(loss = costum_loss1(critic_predictions, cost_v), optimizer = AdamOpt_critic)

    model_actor = Model(inputs = [input_data, f], outputs = poss)  
    model_actor.compile(loss = costum_loss(reward_baseline, log_s), optimizer = AdamOpt_actor)

    ##################### Evaluation dataset ######################
    data = data(config.nTestMats, config.nCells, config.nMuts, config.ms_dir, config.fp, config.fn)
    print('Dataset was created!')

    ################# Loading weights from disk ###################
    model_actor.load_weights(config.restore_from + "/actor.h5")
    print("Weights were loaded to the model!")

    ##################### Error Correction ######################
    solve(model_actor, config, config.input_dimension*config.hidden_dim, data)