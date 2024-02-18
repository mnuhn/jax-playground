import flax.linen as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import flax
import argparse
import numpy as np
import jax
from jax import lax
import jax.nn
from jax import grad, jit, random
import jax.numpy as jnp
import optax
from tqdm import tqdm
import datalib
import lstm
import os

p = argparse.ArgumentParser(description='...')
p.add_argument('--data', type=str)
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--timeseries', type=str)
p.add_argument('--batch_norm', type=bool, default=False)
p.add_argument('--history', type=int, default=65)
p.add_argument('--predictions', type=int, default=12)
p.add_argument('--dense_size', type=int, default=100)
p.add_argument(
    '--features',
    type=str,
    default="wind_speed,gust_speed,wind_dir,air_pressure,air_temp,water_temp")
p.add_argument('--num', type=int, default=1000)
p.add_argument('--model', type=str, default=None)
p.add_argument('--model_file', type=str, default=None)
p.add_argument('--model_arch', type=str, default=None)
p.add_argument('--nonconv_features', type=int, default=0)
p.add_argument('--prediction_file', type=str, default=None)
p = p.parse_args()

with np.load(p.data, mmap_mode=None) as data:
  X = data['x_train']
  Y = data['y_train']

  if len(Y.shape) < 3:
    # Ensure 1 separate dimension for the to-be-predicted features.
    YT = np.expand_dims(YT, 2)
    Y = np.expand_dims(Y, 2)

  XT = data['x_test'][:10000, :, :]
  YT = data['y_test'][:10000, :, :]

  history = X.shape[1]
  predictions = Y.shape[1]

history_len = X.shape[1]
history_feature_cnt = X.shape[2]
predict_len = Y.shape[1]
predict_feature_cnt = Y.shape[2]

print("Loading data from", p.data, "done")

m = None
if p.model == "lstm":
  model_arch = lstm.parse_arch(p.model_arch)
  assert len(model_arch) == 2
  m = lstm.LSTM(
      model_arch=model_arch,
      predictions=predictions,
      features_per_prediction=predict_feature_cnt,
      dropout=0.0,
      nonlstm_features=p.nonconv_features,
      batch_norm=p.batch_norm,
  )

assert m

batcher = datalib.getbatch(X, Y, p.batch_size)
x_batch, y_batch = next(batcher)
x_batch = x_batch[:, :, :]

root_key = jax.random.key(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

variables = m.init(params_key, x_batch, train=False)
params = variables['params']

with open(p.model_file, "rb") as f:
  params = flax.serialization.from_bytes(params, f.read())

pred = m.apply({'params': params}, XT[:, :, :], train=False)

os.makedirs(f'./{p.model_file}.png/', exist_ok=True)

last_predictions = []

all_predictions = []

for i in tqdm(range(0, p.num, p.batch_size)):
  batch_history = np.squeeze(X[i:i + p.batch_size, :, 0])
  batch_features = X[i:i + p.batch_size, :, :]
  batch_prediction = np.squeeze(
      m.apply({'params': params}, batch_features, train=False))
  all_predictions.append(batch_prediction)

all_predictions = np.concatenate(all_predictions)
np.save(p.prediction_file, all_predictions)
