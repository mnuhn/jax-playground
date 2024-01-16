'Train LeNet on MNIST "from scratch" using JAX.'
import argparse
import numpy as np
from tqdm import tqdm

import optax
from jax import grad, jit, random
import jax.numpy as jnp
import jax.nn as nn
import jax
from jax import lax

from helpers import reader
from helpers import eval_print

TRAIN_IMGS='./data/train-images-idx3-ubyte.gz'
TRAIN_LABELS='./data/train-labels-idx1-ubyte.gz'
TEST_IMGS='./data/t10k-images-idx3-ubyte.gz'
TEST_LABELS='./data/t10k-labels-idx1-ubyte.gz'

p = argparse.ArgumentParser(description='...')
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--optimizer',
               choices=['sgd', 'adam', 'self_sgd'], default='adam')
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--bias_term', type=bool, default=True)
p.add_argument('--dense_layers', type=str, default='120,84,10')
p.add_argument('--epochs', type=float, default=300)

args = p.parse_args()
print('args:', args)

print('JAX devices:', jax.devices())
backend = 'cpu'

train_imgs, train_labels = reader.get_data( TRAIN_IMGS, TRAIN_LABELS) # BxHxW
test_imgs, test_labels = reader.get_data( TEST_IMGS, TEST_LABELS) # BxHxW

# Place on device.
train_imgs = jnp.array(train_imgs)
train_labels = jnp.array(train_labels)
test_imgs = jnp.array(test_imgs)
test_labels = jnp.array(test_labels)
num_batches = int(train_imgs.shape[0]/args.batch_size)

def datastream(batch_size):
  global train_imgs, train_labels
  while True:
    key = random.PRNGKey(0)
    perm = jax.random.permutation(key, train_imgs.shape[0])
    train_imgs = train_imgs[perm]
    train_labels = train_labels[perm]
    # Note: We permute the data once every epoch in memory, and then only serve
    # contiguous batches from this. This looked much faster in some of my tests
    # - but I didn't check thoroughly to be really certain.
    for cur_batch in range(0, num_batches):
      yield (
              train_imgs[cur_batch*batch_size:(cur_batch+1)*batch_size],
              train_labels[cur_batch*batch_size:(cur_batch+1)*batch_size])

class LeNet():
  '''LeNet.

     Layer                                        | Dimensions
     ---------------------------------------------|-----------
     Input                                        | Bx28x28x01
     1st 5x5 convolution with padding, 6 channels | Bx28x28x06
     Average pooling                              | Bx14x14x16
     2nd 5x5 convolution no padding, 16 channels  | Bx10x10x16
     Average pooling                              | Bx05x05x16
     Reshape                                      | Bx400
     1st dense layer                              | Bx120 (layers[0])
     2st dense layer                              | Bx84 (layers[1])
     Final layer                                  | Bx10 (layers[2])

    Note: In the original paper, there are only a limited number of connections
    in the 2nd convolution. In this implementation, in the 2nd convolution,
    each output channel has access to all input channels.

    Everything is Xavier initialized (i.e. output channels have std-dev=1).
  '''
  def __init__(self, num_pixels, num_classes, dense_layers, bias_term):
    self.num_pixels = num_pixels
    self.num_classes = num_classes
    self.dense_layers = dense_layers
    self.params = []
    self.param_names = []
    self.bias_term = bias_term

    # 5x5 Kernel, 1 input channel, 6 output channels.
    kernel1 = np.random.normal(size=(5,5,1,6), loc=0, scale=1.0/(25.0**0.5))
    self.params.append(kernel1)
    self.param_names.append("kernel1")

    if self.bias_term:
      kernel1_bias = np.zeros(shape=6)
      self.params.append(kernel1_bias)
      self.param_names.append("kernel1_bias")


    # 5x5 Kernel, 6 input channels, 16 output channels.
    kernel2 = np.random.normal(size=(5,5,6,16), loc=0, scale=1.0/(150.0**0.5))
    self.params.append(kernel2)
    self.param_names.append("kernel2")


    if self.bias_term:
      kernel2_bias = np.zeros(shape=16)
      self.params.append(kernel2_bias)
      self.param_names.append("kernel2_bias")

    # 1st dense layer
    layer = np.random.normal(size=(400, self.dense_layers[0]),
                             loc=0,scale=1.0/400**0.5)
    self.params.append(layer)
    self.param_names.append("dense_1")

    if self.bias_term:
      layer = np.zeros(shape=self.dense_layers[0])
      self.params.append(layer)
      self.param_names.append("dense_1_bias")

    # additional layers as specified via params
    for i in range(1, len(dense_layers)):
      # Initialize such that standard deviation on output is 1.
      layer = np.random.normal(size=(self.dense_layers[i-1],
                                     self.dense_layers[i]),
                               loc=0,scale=(1.0/self.dense_layers[i-1])**0.5)
      self.params.append(layer)
      self.param_names.append(f"layer_{i}")
      if self.bias_term:
        layer = np.zeros(shape=self.dense_layers[i])
        self.params.append(layer)
        self.param_names.append(f"layer_{i}_bias")

    # Place on device.
    for i in range(0, len(self.params)):
      self.params[i] = jnp.array(self.params[i])

  # Returns log softmax.
  def predict(self, params, x):
    hists = []
    x = jnp.expand_dims(x, axis=-1) # Bx28x28x1
    x = lax.conv_general_dilated(lhs=x,
                                 rhs=params[0],
                                 window_strides=(1,1),
                                 padding='SAME', # Use padding.
                                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                 lhs_dilation=(1,1),
                                 rhs_dilation=(1,1)) # Bx28x28x6
    hists.append(x)
    # Bias term.
    x = x + params[1]
    hists.append(x)

    # Avg pooling.
    # This tensor product is needed to ensure we keep all 6 channels separate.
    avg_kernel_4d = jnp.einsum('ij,kl',
                               np.array([[0.25,0.25],[0.25,0.25]]),
                               np.identity(6))

    x = lax.conv_general_dilated(lhs=x,
                                 rhs=avg_kernel_4d,
                                 window_strides=(2,2),
                                 padding='VALID', # No padding.
                                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                 lhs_dilation=(1,1),
                                 rhs_dilation=(1,1)) # Bx14x14x6
    x = nn.sigmoid(x)
    hists.append(x)

    # 2nd convolution (without padding).
    x = lax.conv_general_dilated(lhs=x,
                                 rhs=params[2],
                                 window_strides=(1,1),
                                 padding='VALID',
                                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                 lhs_dilation=(1,1),
                                 rhs_dilation=(1,1)) # Bx10x10x16

    hists.append(x)
    # Bias term.
    x = x + params[3]

    # Avg pooling.
    # This tensor product is needed to ensure we keep all 16 channels separate.
    avg_kernel_4d2 = jnp.einsum('ij,kl',
                                np.array([[0.25,0.25],[0.25,0.25]]),
                                np.identity(16)) # 2x2x16x16

    # 2nd avg pooling.
    x =  lax.conv_general_dilated(lhs=x,
                                  rhs=avg_kernel_4d2,
                                  window_strides=(2,2),
                                  padding='VALID',
                                  dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                  lhs_dilation=(1,1),
                                  rhs_dilation=(1,1)) # Bx5x5x16

    # Flatten.
    x = x.reshape(x.shape[0], -1) # Bx400

    if self.bias_term:
      first_dense_idx = 4
    else:
      first_dense_idx = 2

    for i in range(0, len(self.dense_layers)):
      x = nn.sigmoid(x)
      hists.append(x)
      if self.bias_term:
        x = jnp.einsum('bi,ij',x,params[first_dense_idx+i*2])
        hists.append(x)
        x = x + params[first_dense_idx+i*2+1]
      else:
        x = jnp.einsum('bi,ij',x,params[first_dense_idx+i])

    x = nn.log_softmax(x)
    hists.append(x)
    return x, hists

  def loss(self, params, batch_images,batch_labels):
    p, h = self.predict(params,batch_images)
    r = - jnp.mean(batch_labels * p)
    return r

  def test_loss(self, params):
    p, h = self.predict(params,test_imgs)
    r = - jnp.mean(test_labels * p)
    return r, h

def histo(val):
  val = round(val)
  val = min(max(0,val),7)
  chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"]
  return chars[val]

def train():
  dense_layers = [int(x) for x in args.dense_layers.split(',')]

  model = LeNet(train_imgs.shape[1], train_labels.shape[1],
                     dense_layers=dense_layers, bias_term=args.bias_term)

  model.predict = jit(model.predict, backend=backend)
  model.loss = jit(model.loss, backend=backend)
  model.test_loss = jit(model.test_loss, backend=backend)
  model.gradient = jit(grad(model.loss, argnums=0), backend=backend)

  batches = datastream(batch_size=args.batch_size)

  if args.optimizer == 'sgd':
    tx = optax.sgd(learning_rate=args.learning_rate)
    opt_state = tx.init(model.params)
  elif args.optimizer == 'adam':
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(model.params)

  iterations = int(train_imgs.shape[0] / args.batch_size * args.epochs)

  pbar = tqdm(range(iterations))
  cur_loss = float('nan')
  cur_test_loss = float('nan')
  epoch = 0.0
  for i in pbar:
    cur_images, cur_labels = next(batches)
    g = model.gradient(model.params,cur_images,cur_labels)


    updates, opt_state = tx.update(g, opt_state)
    model.params = optax.apply_updates(model.params, updates)

    epoch = pbar.n * args.batch_size / train_imgs.shape[0]
    pbar.set_description(
            f'epoch={epoch:.3f} train_loss={cur_loss:.3f} ' +
            f'test_loss={cur_test_loss:.3f}')

    if i % int(1+iterations/100) == 0:
      print("Histograms:")
      for p,n in zip(model.params, model.param_names):
        hist, boundaries = np.histogram(p.reshape(-1), bins=100, range=(-1,1), density=True)
        print(f"{n:<25}", "[" + "".join(histo(h) for h in hist) + "]")

      # Print kernels.
      # print(np.moveaxis(model.params[0],[0,1,2],[1,2,0]))
      cur_loss = model.loss(model.params,cur_images,cur_labels)
      cur_test_loss, hists = model.test_loss(model.params)

      print("activations")
      for p,n in zip(hists, model.param_names):

        hist, boundaries = np.histogram(p.reshape(-1), bins=100, range=(0,1), density=True)
        print(f"{n:<25}", "[" + "".join(histo(h) for h in hist) + "]")


      cur_preds = np.argmax(
              model.predict(model.params,test_imgs)[0], axis=-1).tolist()
      cur_labels = np.argmax(test_labels, axis=-1).tolist()
      eval_print.print_confusions(
              i, cur_preds, cur_labels, test_imgs.shape[0], cur_loss)


if __name__ == '__main__':
  train()
