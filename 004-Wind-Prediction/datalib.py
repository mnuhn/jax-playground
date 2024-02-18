import numpy as np
import jax.numpy as jnp
import datetime
from tqdm import tqdm

# TODO:
# * Docstrings
# * Comments
# * Variable Names
# * Break down functions

FIELDS = [
    "year", "month", "day", "hour", "minute", "seconds", "air_temp", "humidity",
    "gust_speed", "wind_speed", "wind_strength", "wind_dir", "wind_chill",
    "water_temp", "air_pressure", "dew_point"
]
FIELDS.extend(["sin_hour", "cos_hour", "sin_wind_dir", "cos_wind_dir"])


def range_to_unit(data, y_min, y_max):
  assert y_min < y_max
  scaled = (data - y_min) / (y_max - y_min)
  scaled = np.clip(scaled, 0.0, 1.0)
  return scaled


def get_data(fn,
             column_names,
             history,
             predictions,
             split_frac=0.8,
             permute=True):
  data = np.load(fn)

  # Windspeed to be predicted.
  column_names = ["wind_speed"] + column_names

  if np.any(np.isnan(data)):
    print("data has nan")

  data_orig = data
  data = preprocess.preprocess_features(data)

  # The first column is the one to be predicted.
  columns = [FIELDS.index(i) for i in column_names]

  data = data[:, columns]
  data_size = len(data)

  # Pre-allocate data.
  X_ALL = np.zeros((data_size - history, history, len(columns) - 1))
  Y_ALL = np.zeros((data_size - history, predictions))

  # Find gaps in data
  skipped_entries = 0

  datetimes = np.array([
      f"{int(year)}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
      for year, month, day, hour, minute, second in data_orig[:, :6]
  ],
                       dtype='datetime64[m]')

  # Compute all positions where data is not 10m apart.
  gaps = np.diff(datetimes) != np.timedelta64(10, 'm')
  gaps = np.insert(gaps, 0, False)
  print("num gaps ", np.sum(gaps))

  idxs_to_delete = []
  for i in tqdm(range(history, data_size - predictions)):
    if np.sum(gaps[i - history:i]) > 0:
      idxs_to_delete.append(i)

    # Skip entries for which there is a gap in time
    X_ALL[i - history, :] = data[i - history:i, 1:]
    Y_ALL[i - history, :] = np.squeeze(data[i:i + predictions,
                                            0])  # just windspeed

  # Delete entries with gaps.
  print("Prior to deletions", X_ALL.shape, Y_ALL.shape)
  X_ALL = np.delete(X_ALL, idxs_to_delete, axis=0)
  Y_ALL = np.delete(Y_ALL, idxs_to_delete, axis=0)
  print("After deletions", X_ALL.shape, Y_ALL.shape)

  if permute:
    np.random.seed(0)
    perm = np.random.permutation(X_ALL.shape[0])
    X_ALL_PERM = X_ALL[perm]
    Y_ALL_PERM = Y_ALL[perm]
  else:
    X_ALL_PERM = X_ALL
    Y_ALL_PERM = Y_ALL

  print("skipped entries", skipped_entries)

  SPLIT = int(data.shape[0] * split_frac)

  X = X_ALL_PERM[:SPLIT]
  Y = Y_ALL_PERM[:SPLIT]

  XT = X_ALL_PERM[SPLIT:]
  YT = Y_ALL_PERM[SPLIT:]

  return X, Y, XT, YT


def getbatch(X, Y, batch_size):
  while True:
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    for i in range(0, int(len(X) / batch_size)):
      yield X[i * batch_size:(i + 1) *
              batch_size, :], Y[i * batch_size:(i + 1) * batch_size]
