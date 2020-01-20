#!

from data import *
from config import get_config, print_config
from predict import solve
from keras.utils import CustomObjectScope
from utils import *
from keras.models import model_from_json

config, _ = get_config()
print_config()


data = data(config.nTestMats, config.nCells, config.nMuts, config.ms_dir, config.fp, config.fn)
print('Dataset was created!')

with CustomObjectScope({'Simple': Simple, 'Decode': Decode, 'StopGrad' : StopGrad, 'Cost' : Cost, 'AttentionLayer': AttentionLayer}):
    json_file = open(config.restore_from + '/actor.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.restore_from + "/actor.h5")
    print("Loaded model from disk")

solve(loaded_model, config, config.input_dimension*config.hidden_dim, data)