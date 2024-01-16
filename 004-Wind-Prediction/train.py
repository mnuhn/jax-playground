from helper import convolution
from jax import grad, jit, random
from jax import lax
from tqdm import tqdm
import argparse
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import flax

import data
import model

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
p.add_argument('--model_name', type=str, default=None)
p = p.parse_args()

def params_debug_str():
  debug_strs = []
  for arg in vars(p):
    value = getattr(p, arg)
    debug_strs.append(f"{arg}={value}")
  debug_str = " ".join(debug_strs)
  return "run: " + debug_str

debug_str = params_debug_str()
print(debug_str)

X, Y, XT, YT = data.get_data(p.timeseries, p.features.split(","), history=p.history, predictions=p.predictions)

m = model.CNN(
        channels=p.channels,
        conv_len=p.conv_len,
        dense_size=p.dense_size,
        down_scale=p.down_scale,
        predictions=p.predictions,
        )

batcher = data.getbatch(X,Y,p.batch_size)
x_batch, y_batch = next(batcher)
params = m.init(jax.random.key(0), x_batch)

value_and_grad = jit(jax.value_and_grad(m.loss))

tx = optax.adam(learning_rate=p.learning_rate)
opt_state = tx.init(params)

pbar = tqdm(range(int(p.iters)))
epoch=0.0

every_iters = int(p.iters / 100.0 * p.debug_every_percent)

for i in pbar:
  if i > p.iters:
    break
  (x, y) = next(batcher)
  cur_loss, g = value_and_grad(params, x, y)
  updates, opt_state = tx.update(g, opt_state)
  params = optax.apply_updates(params, updates)

  if i % every_iters == 0:
    pred = m.apply(params,XT)
    print("Preds", pred)
    print("GTs", YT)
    cur_test_loss = model.loss_pred(pred, YT)
    print("Predictions:", np.min(pred), np.mean(pred), np.max(pred))
    print("GT:", np.min(YT), np.mean(YT), np.max(YT))
    print(debug_str, f"iter={i} test_loss={cur_test_loss:.4f}")

  pbar.set_description(
          f'epoch={epoch:.2f} train_loss={cur_loss:.4f} ' +
          f'test_loss={cur_test_loss:.4f}')

cur_test_loss = m.loss(params,XT, YT)

print(debug_str, f"iter={i} test_loss={cur_test_loss:.4f}")

if p.model_name:
  with open(p.model_name, "wb") as store:
    store.write(flax.serialization.to_bytes(params))

print(debug_str, f"iter={i} test_loss={cur_test_loss:.4f}")
