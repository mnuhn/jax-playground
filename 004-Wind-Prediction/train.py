import os

# Force CPU
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

from flax import metrics
from flax.metrics import tensorboard
from flax.training import train_state
from jax import grad, jit, random
from jax import lax
import flax
import jax
import jax.nn
import jax.numpy as jnp
import jaxopt
import jax.tree_util
import numpy as np
import optax
import tensorflow as tf

from visualizer import Visualizer
from helper import convolution
from tqdm import tqdm
import argparse
import re
import sys

import data
import model

p = argparse.ArgumentParser(description='...')

# Logging.
p.add_argument('--prefix', type=str, default="")
p.add_argument('--log_dir', type=str, default=None)
p.add_argument('--dry_run', type=bool, default=False)
p.add_argument('--debug_every_percent', type=float, default=2.0)
p.add_argument('--draw_every_percent', type=float, default=10.0)
p.add_argument('--tensorboard', type=bool, default=False)
p.add_argument('--png', type=bool, default=False)
p.add_argument('--draw', type=bool, default=False)
p.add_argument('--data', type=str)

# Model params.
p.add_argument('--model', type=str, default=None)
p.add_argument('--down_scale', type=int, default=2)
p.add_argument('--dense_sizes', type=str, default="100")
p.add_argument('--conv_channels', type=str, default="20,40")
p.add_argument('--conv_len', type=int, default=8)
p.add_argument('--nonconv_features', type=int, default=0)
p.add_argument('--padding', type=str, default='SAME')

# Training params.
p.add_argument('--batch_size', type=int, default=256)
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--batch_norm', type=bool, default=False)
p.add_argument('--epochs', type=float, default=10)

# Loss params.
p.add_argument('--loss_fac', type=float, default=1.0)

p.add_argument('--model_name', type=str, default=None)
p = p.parse_args()

def params_debug_str(history_len, history_feature_cnt):
  debug_strs = [f"histlen{history_len}", f"histfeats{history_feature_cnt}"]
  for arg in vars(p):
    if arg == 'data' or arg == 'debug_every_percent' or arg == 'log_dir' or arg == 'model_name' or arg == 'dry_run' or arg == 'tensorboard':
      continue
    value = getattr(p, arg)
    value = re.sub(r'[,]+', '-', str(value))
    debug_strs.append(f"{arg}{value}")
  debug_str = "-".join(debug_strs)
  return debug_str

print("Loading data from", p.data)

with np.load(p.data, mmap_mode=None) as data:
  X = data['x_train']
  Y = data['y_train']

  if len(Y.shape) < 3:
    # Ensure 1 separate dimension for the to-be-predicted features.
    YT = np.expand_dims(YT, 2)
    Y = np.expand_dims(Y, 2)

  XT = data['x_test'][:5000,:,:]
  YT = data['y_test'][:5000,:,:]

  history = X.shape[1]
  predictions = Y.shape[1]

history_len = X.shape[1]
history_feature_cnt = X.shape[2]
predict_len = Y.shape[1]
predict_feature_cnt = Y.shape[2]

print("Loading data from", p.data, "done")

if p.conv_channels == 'None':
  conv_channels = []
else:
  conv_channels = [ int(x) for x in p.conv_channels.split(",") ]

dense_sizes = [ int(x) for x in p.dense_sizes.split(",") ]

if p.log_dir == None:
  debug_str = params_debug_str(history_len, history_feature_cnt)
  p.log_dir = f'tb/{debug_str}'

print("Logging to", p.log_dir)

if os.path.exists(p.log_dir):
  print("Logging dir already exists. Stopping")
  sys.exit(1)

if p.dry_run:
  print("Dry run. Stopping.")
  sys.exit(1)

print('JAX devices:', jax.devices())

m = None
if p.model == "cnn":
  m = model.CNN(
          conv_channels=conv_channels,
          conv_len=p.conv_len,
          dense_sizes=dense_sizes,
          down_scale=p.down_scale,
          predictions=predictions,
          features_per_prediction=predict_feature_cnt,
          batch_norm=p.batch_norm,
          dropout=p.dropout,
          padding=p.padding,
          nonconv_features=p.nonconv_features,
          )
elif p.model == "lstm":
  m = model.LSTM(
          conv_channels=conv_channels,
          dense_sizes=dense_sizes,
          predictions=predictions,
          features_per_prediction=predict_feature_cnt,
          down_scale=p.down_scale,
          dropout=p.dropout,
          nonlstm_features=p.nonconv_features,
          batch_norm=p.batch_norm,
          )

assert m

def getbatch(X,Y,batch_size):
  while True:
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    for i in range(0, int(len(X)/batch_size)):
      yield X[i*batch_size:(i+1)*batch_size,:], Y[i*batch_size:(i+1)*batch_size]

batcher = getbatch(X,Y,p.batch_size)
x_batch, y_batch = next(batcher)

root_key = jax.random.key(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
variables = m.init(params_key, x_batch, train=False)

params = variables['params']
if p.batch_norm:
  batch_stats = variables['batch_stats']
else:
  batch_stats = None

summary_writer = None

if p.tensorboard:
  summary_writer = tensorboard.SummaryWriter(p.log_dir)
else:
  summary_writer = None

class TrainState(train_state.TrainState):
  batch_stats: any

state = TrainState.create(
  apply_fn = m.apply,
  params = params,
  batch_stats = batch_stats,
  tx = optax.adam(learning_rate=p.lr)
)

def scaled_loss(preds, y, non_first_fac):
  loss = (preds-y)**2
  #loss = jaxopt.loss.huber_loss(target=y, pred=preds, delta=0.2)
  # Downscale weight of non-primary features.
  loss = loss.at[:,:,1:].set(loss[:,:,1:]*non_first_fac)
  loss = jnp.mean(jnp.sum(loss,axis=2))
  return loss

@jax.jit
def train_step(state: TrainState, x, y):
  def loss_fn(params):
    if p.batch_norm and p.dropout > 0.0:
      preds, updates = state.apply_fn({
          'params': params,
          'batch_stats': state.batch_stats,
          }, x,train=True, rngs={'dropout': dropout_key}, mutable=['batch_stats'])
      loss = scaled_loss(preds, y, p.loss_fac)
      return loss, (preds, updates)
    elif p.batch_norm:
      preds, updates = state.apply_fn({
          'params': params,
          'batch_stats': state.batch_stats,
          }, x,train=True, mutable=['batch_stats'])
      loss = scaled_loss(preds, y, p.loss_fac)
      return loss, (preds, updates)
    elif p.dropout > 0.0:
      preds = state.apply_fn({
          'params': params,
          }, x,train=True, rngs={'dropout': dropout_key})
      loss = scaled_loss(preds, y, p.loss_fac)
      return loss, (preds, )
    else:
      preds = state.apply_fn({'params': params}, x,train=True)
      loss = scaled_loss(preds, y, p.loss_fac)
      return loss, (preds,)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  if p.batch_norm:
    (cur_train_loss, (preds, updates)), grads = grad_fn(state.params)
  else:
    (cur_train_loss, (preds,)), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  if p.batch_norm:
    state = state.replace(batch_stats=updates['batch_stats'])
  return state, cur_train_loss, grads

#@jax.jit
def eval_step(state: TrainState, x, y):
  if p.batch_norm:
    preds, debug = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x=x, train=False, debug=True)
  else:
    preds, debug = state.apply_fn({'params': state.params}, x=x, train=False, debug=True)

  loss = jnp.mean((preds[:,:,0]-y[:,:,0])**2)
  return state, loss, debug

examples = X.shape[0]
iters = int(p.epochs * (examples / p.batch_size))

print(f"Having {examples} examples and batch size {p.batch_size},")
print(f"to run {p.epochs} epochs need to run {iters}.")

pbar = tqdm(range(int(iters)))

every_iters = int(iters / 100.0 * p.debug_every_percent)
draw_every_iters = int(iters / 100.0 * p.draw_every_percent)

eval_loss = 0.0

if summary_writer:
  summary_writer.hparams(hparams = {
      'batch_size': int(p.batch_size),
      'history_feature_cnt': int(history_feature_cnt),
      'history_len': int(history_len),
      'predict_len': int(predict_len),
      'batch_norm': bool(p.batch_norm),
      'dense_size': str(dense_sizes),
      'conv_channels': str(conv_channels),
      'learning_rate': p.lr,
      'down_scale': int(p.down_scale),
      'dropout': float(p.dropout),
      'conv_len': int(p.conv_len),
      'model': p.model,
      })

for step in pbar:
  if step > iters:
    break
  (x, y) = next(batcher)
  state, train_loss, grads = train_step(state, x, y)
  train_loss = train_loss ** 0.5

  if step > 0 and step % every_iters == 0:
    _, eval_loss, acts = eval_step(state, XT, YT)
    eval_loss = eval_loss ** 0.5
    if summary_writer:
      summary_writer.scalar('eval_loss', eval_loss, step)

      grads_flat, _ = jax.tree_util.tree_flatten_with_path(grads)
      for key_path, value in grads_flat:
        summary_writer.histogram(f"zzz-debug:Gradient{jax.tree_util.keystr(key_path)}",  value, step)

      acts_flat, _ = jax.tree_util.tree_flatten_with_path(acts)
      for key_path, value in acts_flat:
        summary_writer.histogram(f"zzz-debug:Activation{jax.tree_util.keystr(key_path)}",  value, step)

    if step > 0 and p.draw and step % draw_every_iters == 0:
      images = []
      for j in range(0,10):
        v = Visualizer()
        acts["truth"] = YT
        v.draw_dict(acts, num=j)
        if p.png:
          v.save(f"./png/{p.prefix}-test{i:05d}-{j:03d}.png")
        image = tf.image.decode_png(v.byte_array(), channels=4)
        images.append(image)
        del v
      if summary_writer:
        summary_writer.image('zzz-debug:Activations', images, step=step)
        summary_writer.flush()

  if summary_writer:
    summary_writer.scalar('train_loss', train_loss, step)
  pbar.set_description(f"train_loss={train_loss:.06f} eval_loss={eval_loss:.06f}")

if summary_writer:
  summary_writer.flush()

#if p.model_name:
#  with open(p.model_name, "wb") as store:
#    store.write(flax.serialization.to_bytes(params))
