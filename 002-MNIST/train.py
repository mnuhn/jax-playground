'Display the contents of the MNIST dataset with minimal boilerplate.'
import argparse
import numpy as np

from jax import grad, jit, random
import jax.numpy as jnp
import jax.nn as nn
import jax

from helpers import reader
from helpers import train_loop
from helpers import optimize

TRAIN_IMAGES='./data/train-images-idx3-ubyte.gz'
TRAIN_LABELS='./data/train-labels-idx1-ubyte.gz'
TEST_IMAGES='./data/t10k-images-idx3-ubyte.gz'
TEST_LABELS='./data/t10k-labels-idx1-ubyte.gz'

NUM_ROWS = 28
NUM_COLS = 28

p = argparse.ArgumentParser(description='...')
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--optimizer',
               choices=['sgd', 'adam', 'self_sgd'], default='adam')
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--learning_rate_decay', type=float, default=0.01)
p.add_argument('--learning_rate_decay_epoch_step', type=int, default=10)
p.add_argument('--bias_term', type=bool, default=True)
p.add_argument('--layers', type=str, default='120,84,10')
p.add_argument('--epochs', type=float, default=300)

args = p.parse_args()
print('args:', args)

print('JAX devices:', jax.devices())
backend = 'cpu'

train_images, train_labels = reader.get_data(
        TRAIN_IMAGES, TRAIN_LABELS) # (Images, Rows, Cols)
test_images, test_labels = reader.get_data(
        TEST_IMAGES, TEST_LABELS) # (Images, Rows, Cols)

# Place on device.
train_images = jnp.array(train_images)
train_labels = jnp.array(train_labels)
test_images = jnp.array(test_images)
test_labels = jnp.array(test_labels)
num_batches = int(train_images.shape[0]/args.batch_size)

def datastream(batch_size):
  global num_batches
  global train_images, train_labels
  while True:
    key = random.PRNGKey(0)
    perm = jax.random.permutation(key, train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    # Note: We permute the data once every epoch in memory, and then only serve
    # contiguous batches from this. This looked much faster in some of my tests
    # - but I didn't check thoroughly to be really certain.
    for i in range(0, num_batches):
      yield (
              train_images[i*batch_size:(i+1)*batch_size],
              train_labels[i*batch_size:(i+1)*batch_size])

class DenseModel():
  '''Simple dense model.'''
  def __init__(self, num_pixels, num_classes, layers, bias_term):
    self.num_pixels = num_pixels
    self.num_classes = num_classes
    self.layers = layers
    self.params = []
    self.bias_term = bias_term

    # Initialize such that standard deviation on output is 1.
    layer = np.random.normal(size=(NUM_ROWS, NUM_COLS, self.layers[0]),
                             loc=0,scale=1.0/num_pixels**0.5)
    self.params.append(layer)

    if self.bias_term:
      layer = np.zeros(shape=self.layers[0])
      self.params.append(layer)

    for i in range(1, len(layers)):
      # Initialize such that standard deviation on output is 1.
      layer = np.random.normal(size=(self.layers[i-1], self.layers[i]),
                               loc=0,scale=(1.0/self.layers[i-1])**0.5)
      self.params.append(layer)
      if self.bias_term:
        layer = np.zeros(shape=self.layers[i])
        self.params.append(layer)

    for i in range(0, len(self.params)):
      self.params[i] = jnp.array(self.params[i])

  # Returns log softmax.
  def predict(self, params, x):
    if not self.bias_term:
      a = nn.relu(jnp.einsum('xyi,bxy',params[0],x))
      for i in range(1, len(self.layers) - 1):
        a = nn.relu(jnp.einsum('ij,bi',params[i],a))
      a = nn.log_softmax(jnp.einsum('ic,bi',params[-1],a))
    else:
      a = nn.relu(jnp.einsum('xyi,bxy',params[0],x) + params[1])
      for i in range(1, len(self.layers) - 1):
        a = nn.relu(jnp.einsum('ij,bi',params[i*2],a) + params[i*2+1])
      a = nn.log_softmax(jnp.einsum('ic,bi',params[-2],a) + params[-1])

    return a

  def print(self):
    print('Model dump:')
    for cur_param in self.params:
      print('* ', cur_param.shape)
      print(cur_param)
    print()


  def loss(self, params, batch_images,batch_labels):
    return - jnp.mean(batch_labels * self.predict(params,batch_images))

  def test_loss(self, params):
    return - jnp.mean(test_labels * self.predict(params,test_images))

layers = [int(x) for x in args.layers.split(',')]

model = DenseModel(train_images.shape[1], train_labels.shape[1],
                   layers=layers, bias_term=args.bias_term)

predict = jit(model.predict, backend=backend)

model.loss = jit(model.loss, backend=backend)
model.test_loss = jit(model.test_loss, backend=backend)
model.gradient = jit(grad(model.loss, argnums=0), backend=backend)

batches = datastream(batch_size=args.batch_size)

if args.optimizer == 'sgd':
  opt = optimize.OptaxSgd(model, learning_rate=args.learning_rate)
elif args.optimizer == 'adam':
  opt = optimize.OptaxAdam(model, learning_rate=args.learning_rate)
elif args.optimizer == 'self_sgd':
  decay_step = args.learning_rate_decay_epoch_step*50000/args.batch_size
  opt = optimize.SelfSgd(model, learning_rate=args.learning_rate, beta=0.0,
                         decay=args.learning_rate_decay,
                         decay_step=int(decay_step))

iterations = int(train_images.shape[0] / args.batch_size * args.epochs)
train_loop.train_loop(model, batches, test_images, test_labels,
                      opt.update, iterations=iterations)
