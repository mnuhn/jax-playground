from helper import convolution
from jax import grad, jit, random
from jax import lax
from tqdm import tqdm
from flax import metrics
import argparse
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training import train_state
from flax.metrics import tensorboard

import data
import model

p = argparse.ArgumentParser(description='...')
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--channels', type=int, default=20)
p.add_argument('--num_convs', type=int, default=2)
p.add_argument('--conv_len', type=int, default=8)
p.add_argument('--down_scale', type=int, default=2)
p.add_argument('--batch_norm', type=bool, default=False)
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--num_dense', type=int, default=1)
p.add_argument('--dense_size', type=int, default=100)
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--iters', type=float, default=150000)
p.add_argument('--data', type=str)
p.add_argument('--debug_every_percent', type=int, default=1)
p.add_argument('--model', type=str, default=None)
p.add_argument('--model_name', type=str, default=None)
p.add_argument('--log_dir', type=str, default="./tensorboard/default")
p = p.parse_args()

def params_debug_str():
  debug_strs = []
  for arg in vars(p):
    value = getattr(p, arg)
    debug_strs.append(f"{arg}={value}")
  debug_str = " ".join(debug_strs)
  return debug_str

with np.load(p.data) as data:
  X = data['x_train']
  Y = data['y_train']
  XT = data['x_test']
  YT = data['y_test']

  history = X.shape[1]
  predictions = Y.shape[1]

print(X.shape)
print(Y.shape)

print(XT.shape)
print(YT.shape)

print("batch_norm", p.batch_norm)

m = None
if p.model == "cnn":
  m = model.CNN(
          channels=p.channels,
          conv_len=p.conv_len,
          num_convs=p.num_convs,
          dense_size=p.dense_size,
          num_dense=p.num_dense,
          down_scale=p.down_scale,
          predictions=predictions,
          batch_norm=p.batch_norm,
          dropout=p.dropout,
          )
elif p.model == "lstm":
  m = model.LSTM(
          hidden_state_dim=p.channels,
          dense_size=p.dense_size,
          predictions=predictions,
          )

assert m

def getbatch(X,Y,batch_size):
  while True:
    perm = np.random.permutation(X.shape[0])
    print("permute")
    X = X[perm]
    Y = Y[perm]
    print("permute_done")
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

#for layer_params in params["params"].items():
#  print(f"Layer {layer_params[0]}")
#  for k, v in layer_params[1].items():
#    print(f"  Param {k} ({v.shape})")

#value_and_grad = jit(jax.value_and_grad(m.loss_training))

#opt_state = tx.init(params)

summary_writer = tensorboard.SummaryWriter(p.log_dir)

class TrainState(train_state.TrainState):
  batch_stats: any

state = TrainState.create(
  apply_fn = m.apply,
  params = params,
  batch_stats = batch_stats,
  tx = optax.adam(learning_rate=p.learning_rate)
)

@jax.jit
def train_step(state: TrainState, x, y):
  def loss_fn(params):
    if p.batch_norm and p.dropout > 0.0:
      preds, updates = state.apply_fn({
          'params': params,
          'batch_stats': state.batch_stats,
          }, x,train=True, rngs={'dropout': dropout_key}, mutable=['batch_stats'])
      loss = jnp.mean((preds-y)**2)
      return loss, (preds, updates)
    elif p.batch_norm:
      preds, updates = state.apply_fn({
          'params': params,
          'batch_stats': state.batch_stats,
          }, x,train=True, mutable=['batch_stats'])
      loss = jnp.mean((preds-y)**2)
      return loss, (preds, updates)
    elif p.dropout > 0.0:
      preds = state.apply_fn({
          'params': params,
          }, x,train=True, rngs={'dropout': dropout_key})
      loss = jnp.mean((preds-y)**2)
      return loss, (preds, )
    else:
      preds = state.apply_fn({'params': params}, x,train=True)
      loss = jnp.mean((preds-y)**2)
      return loss, (preds,)

    #jax.debug.print("params={params}", params=params['Dense_1']['kernel'])
    #jax.debug.print("preds={preds}", preds=preds)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  if p.batch_norm:
    (cur_train_loss, (preds, updates)), grads = grad_fn(state.params)
  else:
    (cur_train_loss, (preds,)), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  if p.batch_norm:
    state = state.replace(batch_stats=updates['batch_stats'])
  return state, cur_train_loss

#@jax.jit
def eval_step(state: TrainState, x, y):
  if p.batch_norm:
    preds = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x=x, train=False)
  else:
    preds = state.apply_fn({'params': state.params}, x=x, train=False)

  #jax.debug.print("batch_stats={batch_stats}", batch_stats=state.batch_stats)
  jax.debug.print("x={x}", x=x)
  jax.debug.print("preds={preds}", preds=preds)
  jax.debug.print("y={y}", y=y)
  loss = jnp.mean((preds-y)**2)
  return state, loss

pbar = tqdm(range(int(p.iters)))
epoch=0.0

every_iters = int(p.iters / 100.0 * p.debug_every_percent)

debug_str = params_debug_str()
eval_loss = 0.0

summary_writer.hparams(hparams = {
    'batch_size': int(p.batch_size),
    'batch_norm': bool(p.batch_norm),
    'num_dense': int(p.num_dense),
    'dense_size': int(p.dense_size),
    'learning_rate': p.learning_rate,
    'down_scale': int(p.down_scale),
    'conv_len': int(p.conv_len),
    'channels': int(p.channels),
    'model': p.model,
    })

for i in pbar:
  if i > p.iters:
    break
  (x, y) = next(batcher)
  #print("Train Step")
  #print(state.batch_stats)
  state, train_loss = train_step(state, x, y)
  train_loss = train_loss ** 0.5

  if i % 1000 == 0:
    _, eval_loss = eval_step(state, XT, YT)
    eval_loss = eval_loss ** 0.5
    summary_writer.scalar('eval_loss', eval_loss, i)
    #jax.debug.print("kernel_shape={x}",x=state.params['Conv_0']['kernel'].shape)
    #jax.debug.print("kernel={x}",x=jnp.transpose(state.params['Conv_0']['kernel'],axes=[2,1,0]))

  summary_writer.scalar('train_loss', train_loss, i)
  pbar.set_description(f"train_loss={train_loss:.06f} eval_loss={eval_loss:.06f}")
  #summary_writer.scalar('train_loss', 0.3, 2.0)
summary_writer.flush()

#if p.model_name:
#  with open(p.model_name, "wb") as store:
#    store.write(flax.serialization.to_bytes(params))
