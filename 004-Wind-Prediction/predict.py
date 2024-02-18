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
import model
import datalib
import os

p = argparse.ArgumentParser(description='...')
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--timeseries', type=str)
p.add_argument('--history', type=int, default=65)
p.add_argument('--predictions', type=int, default=12)
p.add_argument('--channels', type=int, default=20)
p.add_argument('--conv_len', type=int, default=8)
p.add_argument('--down_scale', type=int, default=2)
p.add_argument('--dense_size', type=int, default=100)
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--iters', type=float, default=150000)
p.add_argument('--features', type=str, default="wind_speed,gust_speed,wind_dir,air_pressure,air_temp,water_temp")
p.add_argument('--debug_every_percent', type=int, default=1)
p.add_argument('--num', type=int, default=1000)
p.add_argument('--model', type=str, default=None)
p.add_argument('--model_name', type=str, default=None)
p.add_argument('--prediction_file', type=str, default=None)
p = p.parse_args()

X, Y, XT, YT = data.get_data(p.timeseries, ["wind_speed"] + p.features.split(","), history=p.history, predictions=p.predictions, permute=False)

m = None
if p.model == "lstm":
  m = model.LSTM(
          hidden_state_dim=p.channels,
          dense_size=p.dense_size,
          predictions=p.predictions,
          )

assert m


batcher = datalib.getbatch(X,Y,p.batch_size)
x_batch, y_batch = next(batcher)
x_batch = x_batch[:,:,1:]
params = m.init(jax.random.key(0), x_batch)

with open(p.model_name, "rb") as f:
  params = flax.serialization.from_bytes(params, f.read())

pred = m.apply(params, XT[:,:,1:])

os.makedirs(f'./{p.model_name}.png/', exist_ok=True)

last_predictions = []

all_predictions = []

for i in tqdm(range(0,p.num,p.batch_size)):
  batch_history = np.squeeze(X[i:i+p.batch_size, :, 0])
  batch_features = X[i:i+p.batch_size, :, 1:]
  batch_prediction = np.squeeze(m.apply(params, batch_features))

  all_predictions.append(batch_prediction)


  continue

all_predictions = np.concatenate(all_predictions)
np.save(p.prediction_file, all_predictions)
