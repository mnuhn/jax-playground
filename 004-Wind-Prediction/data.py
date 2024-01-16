import numpy as np
import jax.numpy as jnp

FIELDS = ["year", "month", "day", "hour", "minute", "seconds", "air_temp", "humidity", "gust_speed", "wind_speed", "wind_strength", "wind_dir", "wind_chill", "water_temp", "air_pressure", "dew_point"]
FIELDS.extend(["sin_hour", "cos_hour", "sin_wind_dir", "cos_wind_dir"])

def range_to_unit(data, y_min, y_max):
  assert y_min < y_max
  scaled = (data - y_min) / (y_max - y_min)
  scaled = np.clip(scaled, 0.0, 1.0)
  return scaled

def get_data(fn, column_names, history, predictions, split_frac=0.8, permute=True):
  data = np.load(fn)

  # Windspeed to be predicted.
  column_names = ["wind_speed"] + column_names

  if np.any(np.isnan(data)):
    print("data has nan")

  # Virtual indexes, see below.
  data = np.c_[data, data[:,FIELDS.index("hour")]]
  data = np.c_[data, data[:,FIELDS.index("hour")]]
  data = np.c_[data, data[:,FIELDS.index("wind_dir")]]
  data = np.c_[data, data[:,FIELDS.index("wind_dir")]]

  data[:,FIELDS.index("wind_speed")] = range_to_unit(data[:,FIELDS.index("wind_speed")], 0.0, 25.0)
  data[:,FIELDS.index("gust_speed")] = range_to_unit(data[:,FIELDS.index("gust_speed")], 0.0, 25.0)
  data[:,FIELDS.index("air_pressure")] = range_to_unit(data[:,FIELDS.index("air_pressure")], 950.0, 1050.0)
  data[:,FIELDS.index("air_temp")] = range_to_unit(data[:,FIELDS.index("air_temp")], -15.0, 40.0)
  data[:,FIELDS.index("water_temp")] = range_to_unit(data[:,FIELDS.index("water_temp")], 0.0, 30.0)
  data[:,FIELDS.index("minute")] = range_to_unit(data[:,FIELDS.index("minute")], 0.0, 59.0)

  data[:,FIELDS.index("sin_hour")] = jnp.sin(2.0 * np.pi * data[:, FIELDS.index("sin_hour")] / 24.0)
  data[:,FIELDS.index("cos_hour")] = jnp.cos(2.0 * np.pi * data[:, FIELDS.index("cos_hour")] / 24.0)
  data[:,FIELDS.index("hour")] = range_to_unit(data[:,FIELDS.index("hour")], 0.0, 24.0)

  data[:,FIELDS.index("sin_wind_dir")] = jnp.sin(2.0 * np.pi * data[:, FIELDS.index("sin_wind_dir")] / 360.0)
  data[:,FIELDS.index("cos_wind_dir")] = jnp.cos(2.0 * np.pi * data[:, FIELDS.index("cos_wind_dir")] / 360.0)
  data[:,FIELDS.index("wind_dir")] = range_to_unit(data[:,FIELDS.index("hour")], 0.0, 359.0)

  # The first column is the one to be predicted.
  columns=[FIELDS.index(i) for i in column_names]

  data = data[:,columns]

  data_size = len(data)
  X_ALL = np.zeros((data_size - history, history, len(columns)-1))
  Y_ALL = np.zeros((data_size - history, predictions))
  for i in range(history, data_size - predictions):
    X_ALL[i-history,:] = data[i-history:i,1:]
    Y_ALL[i-history,:] = np.squeeze(data[i:i+predictions,0]) # just windspeed

  if permute:
    np.random.seed(0)
    perm = np.random.permutation(X_ALL.shape[0])
    X_ALL_PERM = X_ALL[perm]
    Y_ALL_PERM = Y_ALL[perm]
  else:
    X_ALL_PERM = X_ALL
    Y_ALL_PERM = Y_ALL

  SPLIT = int(data.shape[0] * split_frac)

  X = X_ALL_PERM[:SPLIT]
  Y = Y_ALL_PERM[:SPLIT]

  XT = X_ALL_PERM[SPLIT:]
  YT = Y_ALL_PERM[SPLIT:]

  return X, Y, XT, YT

def getbatch(X,Y,batch_size):
  while True:
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    for i in range(0, int(len(X)/batch_size)):
      yield X[i*batch_size:(i+1)*batch_size,:], Y[i*batch_size:(i+1)*batch_size]
