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
import os
import sys
import jaxopt
import tensorflow as tf
from flax.training import train_state
from flax.metrics import tensorboard
from visualizer import Visualizer

import data
import model

p = argparse.ArgumentParser(description='...')
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--conv_channels', type=str, default="20")
p.add_argument('--loss_fac', type=float, default=1.0)
p.add_argument('--conv_len', type=int, default=8)
p.add_argument('--nonconv_features', type=int, default=0)
p.add_argument('--down_scale', type=int, default=2)
p.add_argument('--batch_norm', type=bool, default=False)
p.add_argument('--tensorboard', type=bool, default=False)
p.add_argument('--draw', type=bool, default=False)
p.add_argument('--png', type=bool, default=False)
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--num_dense', type=int, default=1)
p.add_argument('--padding', type=str, default='VALID')
p.add_argument('--dense_size', type=int, default=100)
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--epochs', type=float, default=10)
p.add_argument('--data', type=str)
p.add_argument('--debug_every_percent', type=int, default=1)
p.add_argument('--model', type=str, default=None)
p.add_argument('--prefix', type=str, default="")
p.add_argument('--model_name', type=str, default=None)
p.add_argument('--log_dir', type=str, default=None)
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

  if len(Y.shape) < 3:
    # Ensure 1 separate dimension for the to-be-predicted features.
    YT = np.expand_dims(YT, 2)
    Y = np.expand_dims(Y, 2)

  history = X.shape[1]
  predictions = Y.shape[1]

history_len = X.shape[1]
history_feature_cnt = X.shape[2]
predict_len = Y.shape[1]
predict_feature_cnt = Y.shape[2]

if p.log_dir == None:
  p.log_dir = f'./tensorboard/{p.prefix}lossfac{p.loss_fac}-historylen{history_len}-historyfeaturecnt{history_feature_cnt}-predictlen{predict_len}-predictfeaturecnt{predict_feature_cnt}model{p.model}-convlen{len(conv_channels)}-dscale{p.down_scale}-chans{p.conv_channels}-padding{p.padding}-densesize{p.dense_size}-numdense{p.num_dense}-lr{p.learning_rate}-bs{p.batch_size}'

print("Logging to", p.log_dir)

if os.path.exists(p.log_dir):
  print("Logging dir already exists. Stopping")
  sys.exit(1)

conv_channels = [ int(x) for x in p.conv_channels.split(",") ]

m = None
if p.model == "cnn":
  m = model.CNN(
          conv_channels=conv_channels,
          conv_len=p.conv_len,
          dense_size=p.dense_size,
          num_dense=p.num_dense,
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
  tx = optax.adam(learning_rate=p.learning_rate)
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
  return state, cur_train_loss, grads

#@jax.jit
def eval_step(state: TrainState, x, y):
  if p.batch_norm:
    preds, debug = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x=x, train=False, debug=True)
  else:
    preds, debug = state.apply_fn({'params': state.params}, x=x, train=False, debug=True)

  #jax.debug.print("batch_stats={batch_stats}", batch_stats=state.batch_stats)
  #jax.debug.print("x={x}", x=x)
  #jax.debug.print("preds={preds}", preds=preds)
  #jax.debug.print("y={y}", y=y)
  loss = jnp.mean((preds[:,:,0]-y[:,:,0])**2)
  return state, loss, debug

examples = X.shape[0]
iters = int(p.epochs * (examples / p.batch_size))

print(f"Having {examples} examples and batch size {p.batch_size},")
print(f"to run {p.epochs} epochs need to run {iters}.")

pbar = tqdm(range(int(iters)))

every_iters = int(iters / 100.0 * p.debug_every_percent)

debug_str = params_debug_str()
eval_loss = 0.0

if summary_writer:
  summary_writer.hparams(hparams = {
      'batch_size': int(p.batch_size),
      'history_feature_cnt': int(history_feature_cnt),
      'history_len': int(history_len),
      'predict_len': int(predict_len),
      'batch_norm': bool(p.batch_norm),
      'num_dense': int(p.num_dense),
      'dense_size': int(p.dense_size),
      'learning_rate': p.learning_rate,
      'down_scale': int(p.down_scale),
      'conv_len': int(p.conv_len),
      # ADD conv channels
      'num_convs': int(len(conv_channels)),
      'model': p.model,
      })

for i in pbar:
  if i > iters:
    break
  (x, y) = next(batcher)
  #print("Train Step")
  #print(state.batch_stats)
  state, train_loss, grads = train_step(state, x, y)
  train_loss = train_loss ** 0.5

  if i % every_iters == 0:
    _, eval_loss, debug = eval_step(state, XT, YT)
    eval_loss = eval_loss ** 0.5
    if summary_writer:
      summary_writer.scalar('eval_loss', eval_loss, i)
    for k in grads.keys():
      for l in grads[k]:
        print(k, l, np.mean(grads[k][l]))

    if p.draw:
      images = []
      for j in range(0,10):
        v = Visualizer()
        debug["truth"] = YT
        v.draw_dict(debug, num=j)
        if p.png:
          v.save(f"./png/{p.prefix}-test{i:05d}-{j:03d}.png")
        image = tf.image.decode_png(v.byte_array(), channels=4)
        images.append(image)
        del v
      if summary_writer:
        summary_writer.image('activations', images, step=i)
        summary_writer.flush()


    #jax.debug.print("kernel_shape={x}",x=state.params['Conv_0']['kernel'].shape)
    #jax.debug.print("kernel={x}",x=jnp.transpose(state.params['Conv_0']['kernel'],axes=[2,1,0]))

  if summary_writer:
    summary_writer.scalar('train_loss', train_loss, i)
  pbar.set_description(f"train_loss={train_loss:.06f} eval_loss={eval_loss:.06f}")

if summary_writer:
  summary_writer.flush()

#if p.model_name:
#  with open(p.model_name, "wb") as store:
#    store.write(flax.serialization.to_bytes(params))
