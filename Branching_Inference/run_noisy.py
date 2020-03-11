import config
from util import *
import tensorflow as tf
from tensorflow import keras


parser = argparse.ArgumentParser(description='Run an experiment for evaluating noise-tolerance')
parser.add_argument('--modelfile', '-m',  type=str, help='model file name')


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
    return np.expand_dims(X, axis=-1), y

def _make_noisy(X_clean, rates=None, ks=None, ):
  X_clean = np.squeeze(X_clean, axis=-1)
  print(X_clean.shape)
  X = np.empty(X_clean.shape)
  for i in range(X_clean.shape[0]):
    X[i] = add_noise(X_clean[i], rates=rates, ks=ks) #reshape(n_cells, n_muts), rates=rates, ks=ks).reshape(n_cells * n_muts)
  print(X.shape)
  return np.expand_dims(X, axis=-1)

#from train import _make_dataset

def run_experiment():
  n_cells, n_muts = 100, 100
  dataset_size = 100
  rows_dict = []
  rates = [0, 0]
  n_datapoint = 50
  factor = 1

  modelAddress = os.path.join(config.modelsDir, args.modelfile)
  model = load_model(modelAddress, custom_objects={"tf":tf, "keras":keras}) #, 'BinaryAccuracy': keras.metrics.binary_accuracy})	

  ##### this is for activating cache
  data_set = _make_dataset(n_cells, n_muts, 1)
  ypredict = model.predict(data_set[0])
  ###########

  for row_ind in tqdm(range(50)):
    rates[1] = 0.35 * row_ind / n_datapoint
    rates[0] = rates[1] * factor

    for n_cells, n_muts in [(100,100),(300,300)]:
      data_set = _make_dataset(n_cells, n_muts, dataset_size)
      X = data_set[0]
      y = data_set[1]
      X_clean, y = shuffle(X, y)
      n = len(y)

      X = _make_noisy(X_clean, rates=rates)

      DL_time = time.time()
      ypredict = model.predict(X).squeeze(axis=-1)
      DL_time = time.time() - DL_time
      #print(y.shape)
      #print(ypredict.shape)
      #predicted_labels = np.argmax(ypredict, axis=1)
      #original_labels = y

      n_correct = np.sum(y == (ypredict > 0.5)) #predicted_labels == original_labels)
      accuracy = n_correct / n
      row = {
        "DL_time": DL_time,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "fp_rate": rates[0],
        "fn_rate": rates[1],
      }
      rows_dict.append(row)

  df = pd.DataFrame(rows_dict)
  csv_file_name = os.path.join(config.outputsDir, f"output_{time.time()}.csv")
  df.to_csv(csv_file_name)
  print(f"csvFile is saved to {csv_file_name}")


if __name__ == '__main__':
  import __main__
  print(f"{__main__.__file__} starts here")
  run_experiment()
