import numpy as np
import jax.numpy as jnp
import datetime
from tqdm import tqdm
import argparse
import os
import preprocess

def file_doesnt_exist(fn):
  if os.path.exists(fn):
    raise argparse.ArgumentTypeError(f"File '{fn}' already exists.")
  return fn

# TODO:
# * Docstrings
# * Comments
# * Variable Names
# * Break down functions

parser = argparse.ArgumentParser(description="Process the location option.")
parser.add_argument("--files", type=str, required=True)
parser.add_argument("--output_file", type=file_doesnt_exist, required=True)
parser.add_argument('--test_year', type=int, default=2023)
parser.add_argument('--history', type=int, default=257)
parser.add_argument('--future', type=int, default=128)
parser.add_argument('--copy', type=bool, default=False)
parser.add_argument('--history_features', type=str, default="wind_speed,gust_speed,air_pressure,air_temp,sin_wind_dir,cos_wind_dir,sin_hour,cos_hour")
parser.add_argument('--predict_features', type=str, default='wind_speed')
args = parser.parse_args()

def generate_examples(fns, history, predictions, history_features, predict_features, test_year=args.test_year, permute=True):
  data = None

  for fn in fns:
    if data is None:
      data = np.load(fn)
    else:
      data = np.concatenate([data, np.load(fn)])

  if np.any(np.isnan(data)):
    print("data has nan")

  # The first column is the one to be predicted.
  history_columns = [preprocess.FIELDS.index(i) for i in history_features]
  predict_columns = [preprocess.FIELDS.index(i) for i in predict_features]

  data_orig = data
  data = preprocess.preprocess_features(data)
  data_size = len(data)

  
  # Pre-allocate data.
  X_ALL = np.zeros((data_size - history, history, data.shape[1]), dtype=np.float32)
  Y_ALL = np.zeros((data_size - history, predictions, data.shape[1]), dtype=np.float32)

  out_idx = 0
  skipped = 0
  for i in tqdm(range(history, data_size - predictions)):
    if np.sum(data[i-history:i, preprocess.GAP]) > 0:
      skipped += 1
      continue

    X_ALL[out_idx,:] = data[i-history:i,:]
    if args.copy:
      print("WARNING: THIS GENERATES DATA WITH OVERLAPPING HISTORY AND PREDS")
      print("WARNING: THIS GENERATES DATA WITH OVERLAPPING HISTORY AND PREDS")
      Y_ALL[out_idx,:] = np.squeeze(data[i-history:i-history+predictions,:]) # just windspeed
    else:
      Y_ALL[out_idx,:] = np.squeeze(data[i:i+predictions,:]) # just windspeed

    out_idx += 1

  deleted = X_ALL.shape[0] - out_idx
  print(f"Skipped {skipped} examples")
  print("Prior to deletions", X_ALL.shape, Y_ALL.shape)

  X_ALL = X_ALL[:out_idx, :, :]
  Y_ALL = Y_ALL[:out_idx, :, :]

  print("After deletions", X_ALL.shape, Y_ALL.shape)
  print("Years:", np.unique(X_ALL[:, :, preprocess.YEAR]))

  test_mask = (np.any(X_ALL[:, :, preprocess.YEAR] == test_year, axis=1))
  train_mask = ~test_mask

  print("#test:", np.sum(test_mask))
  print("#train:", np.sum(train_mask))

  X_ALL = X_ALL[:, :, history_columns]
  Y_ALL = Y_ALL[:, :, predict_columns]
  X  = X_ALL[train_mask, :, :]
  Y  = Y_ALL[train_mask, :, :]
  XT = X_ALL[test_mask, :, :]
  YT = Y_ALL[test_mask, :, :]

  print("Train", X.shape, Y.shape)
  print("Test", XT.shape, YT.shape)

  if permute:
    print("Permuting")
    np.random.seed(0)
    x_perm = np.random.permutation(X.shape[0])
    xt_perm = np.random.permutation(XT.shape[0])
    X = X[x_perm]
    Y = Y[x_perm]
    XT = XT[xt_perm]
    YT = YT[xt_perm]

  np.savez(args.output_file, x_train=X, y_train=Y, x_test=XT, y_test=YT)


generate_examples(args.files.split(","),
                  history=args.history,
                  predictions=args.future,
                  history_features=args.history_features.split(","),
                  predict_features=args.predict_features.split(","))
