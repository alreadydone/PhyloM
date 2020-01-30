from imports import *

from util import isViolated
from util import count3gametes

from dataset import DataSet 

from KerasModels import addDenseLayerClassification

import pandas as pd


class LossAccHistory(keras.callbacks.Callback):
    def __init__(self, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y
        self.test_loss = []
        self.test_acc = []
        self.train_loss = []
        self.train_acc = []

    def on_epoch_end(self, epoch, logs={}):
        te_loss, te_acc = self.model.evaluate(self.test_X, self.test_y, verbose=0)
        self.test_loss.append(te_loss)
        self.test_acc.append(te_acc)
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('binary_accuracy'))


def main():
    config, _ = get_config()
    print_config()

    dataset = DataSet(config)

    if not os.path.isfile(config.h5_dir + "/" + config.h5_filename_train):
        data = dataset.Cat_Shuf(train = True)

        dataset.saveDataSet(data[0], data[1], config.h5_filename_train)

    if not os.path.isfile(config.h5_dir + "/" + config.h5_filename_test):
        data = dataset.Cat_Shuf(train = False)

        dataset.saveDataSet(data[0], data[1], config.h5_filename_test)

    train_matrices, train_labels = dataset.loadDataSet(config.h5_filename_train)
    test_matrices, test_labels = dataset.loadDataSet(config.h5_filename_test)

    models = []
    addDenseLayerClassification(models, inputDim=(config.nCells, config.nMuts), nameSuffix="sg",
                                hiddenSize=100,
                                nLayers=2,
                                dropOutRate=0.2, 
                                dropOutFirst=False,
                                dropOutAfterFirst=True,
                                activation="sigmoid",
                                useSoftmax=False)

    for name, model in models:

        plot_model(model, to_file= config.output_dir + '/{}_plot.png'.format(name), show_shapes=True, show_layer_names=True)
        history = LossAccHistory(test_matrices, test_labels)
        model.fit(train_matrices, train_labels, epochs = config.nb_epoch, verbose = 0, callbacks = [history])

        df = pd.DataFrame(index = ['epoch_{}'.format(e) for e in range(nE)], \
            columns = ['train_acc', 'test_acc', 'train_loss', 'test_loss'])

        df['train_acc'] = history.train_acc
        df['test_acc'] = history.test_acc
        df['train_loss'] = history.train_loss
        df['test_loss'] = history.test_loss
        df.to_csv(config.output_dir + '/{name}_{nCells}x{nMuts}.csv'.format(name = name, nCells = config.nCells, nMuts = config.nMuts), sep = ',')


if __name__ == '__main__':
    main()



