import config
# from tensorflow import keras
import importlib
import util
importlib.reload(util)
from util import *

parser = argparse.ArgumentParser(description='Train model on an input matrix')
parser.add_argument('-n',  type=int, help='num of cells', default=100)
parser.add_argument('-m',  type=int, help='num of muts', default=100)
parser.add_argument('-d',  type=int, help='dataset size', default=5000)
parser.add_argument('-e',  type=int, help='num epochs', default=200)
parser.add_argument('-s',  type=int, help='hidden size', default=100)
args = parser.parse_args()

def _make_dataset(nCells, nMuts, n):  # TODO: the weights of rows might be bias

    X = np.zeros((2 * n, nCells, nMuts), dtype=np.int8)
    for i in range(n):
        res = True
        while res:
            original1 = run_ms("temp.txt", nCells, nMuts)
            res = is_linear(original1)
        original1 = colShuffle(original1)
        X[i] = rowSort(original1)

    for i in range(n):
        w = np.sum(X[i])
        original2 = make_constrained_linear_input(nCells, nMuts, w)
        original2 = colShuffle(original2)
        X[i + n] = rowSort(original2)

    y = np.array([1]*n + [0]*n)
    #return X, y
    #print(X.shape)
    #X = np.expand_dims(X, axis=-1)
    #print(X.shape)
    #print('\n\n')
    return np.expand_dims(X, axis=-1), y

if __name__ == '__main__':
  import __main__
  config.logger.info(f"{__main__.__file__} starts here")
  n_cells = args.n
  n_muts = args.m
  dataset_size = args.d
  valRatio = 0.1
  nEpochs = args.e

  batchSizeList = [None]
  dataSetList = [_make_dataset(n_cells, n_muts, dataset_size)]
  print(dataSetList[0][0].shape)
  print('\n\n')
  #print(dataSetList[0][0])
  df = pd.DataFrame()
  for dataSet in dataSetList:
    if isinstance(dataSet, str):
      dataSetName = dataSet
      inputReadTime = time.time()
      x, y = loadDataSet(dataSetName)
      inputReadTime = time.time() - inputReadTime
      config.logger.info(f"Input is read x.shape={x.shape}, y.shape={y.shape} from {dataSetName}")
      config.logger.info(f"inputReadTime = {inputReadTime}")
    else:
      dataSetName = f"make_dataset({n_cells}, {n_muts}, {dataset_size})"
      x, y = dataSet
      x, y = sk.utils.shuffle(x, y)
      config.logger.info(f"Input is read x.shape={x.shape}, y.shape={y.shape}")

    valSize = int(x.shape[0] * valRatio)

    config.logger.info(f"ysum = {np.sum(y, axis = 0)}")

    models = []
    makeModelsTime = time.time()

    add_col_conv_model(models, inputDims=(n_cells, n_muts), nameSuffix="model_",
                       hiddenSize=4, nLayers=3)

    makeModelsTime = time.time() - makeModelsTime
    config.logger.info(f"number of models:{len(models)}")
    config.logger.info(f"makeModelsTime = {makeModelsTime}")

    # saveModels(models)
    for name, model in tqdm(models):
      print(batchSizeList)
      for batchSize in batchSizeList:
        # model.load_weights(os.path.join(tempDir, f"tempModel_{name}.h5"))
        config.logger.info(f"fit start for {name}")
        # config.logger.info((x.shape, y.shape))
        fitTime = time.time()
        hist = model.fit(
          x,
          y,
          epochs = nEpochs,
          batch_size = batchSize,
          verbose = 0,
          validation_split = valRatio,
          #     validation_freq=10,
        )
        print(hist)
        fitTime = time.time()-fitTime
        config.logger.info("fit finished")
        config.logger.info(f"fitTime = {fitTime}")
        saveTime = time.time()
        histFileName = f"trainingHistory_{name}_{dataSetName}_{batchSize}_{saveTime}"
        with open(os.path.join(config.historiesDir, histFileName), 'wb') as file_pi:
          pickle.dump(hist.history, file_pi)
        config.logger.info(f"The history is saved in {histFileName}")

        modelFileName = os.path.join(config.modelsDir, f"tempModel_fullModel_{nEpochs}_{batchSize}_{name}_{saveTime}.h5")
        keras.models.save_model(model, modelFileName) # model.save(modelFileName)
        saveTime = time.time() - saveTime
        config.logger.info(f"model is saved to {modelFileName}")
        row = {
            "batchSize": batchSize,
            "datasetName": dataSetName,
            "modelFileName": modelFileName,
            "histFileName": histFileName,
            "fitTime": fitTime,
            "saveTime": saveTime,
            "bestTrainAcc": np.max(hist.history["binary_accuracy"]),
            "finalTrainAcc": hist.history["binary_accuracy"][-1],
            "bestValAcc": np.max(hist.history["val_binary_accuracy"]),
            "finalValAcc": hist.history["val_binary_accuracy"][-1],
          }
        df = df.append(row, ignore_index = True)
        m = load_model(modelFileName)
        # config.logger.info(row)
  csvFileName = os.path.join(config.outputsDir, f"output_{time.time()}.csv")
  df.to_csv(csvFileName)
  config.logger.info(f"csvFile is saved to {csvFileName}")

