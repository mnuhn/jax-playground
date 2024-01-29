import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen import initializers
from flax import traverse_util
from jax import tree_util

class LSTM(nn.Module):
  predictions: int
  hidden_state_dim: int
  dense_size: int

  def setup(self):
    pass

  @nn.compact
  def __call__(self, x):
    cell = nn.OptimizedLSTMCell(self.hidden_state_dim)
    def body_fn(cell, carry, x):
      return cell(carry, x)
    scan = nn.scan(
      body_fn, variable_broadcast="params",
      split_rngs={"params": False}, in_axes=1, out_axes=1)

    input_shape =  x[:, 0, :].shape
    carry = cell.initialize_carry(
      jax.random.key(0), input_shape)
    carry, x = scan(cell, carry, x)

    # Take only last hidden state.
    x = x[:,-1,:]
    x = nn.Dense(features=self.dense_size)(x)
    x = nn.sigmoid(x)

    x = nn.Dense(features=self.predictions)(x)
    x = nn.sigmoid(x)
    return x

  def loss(self, params, x, y):
    p = self.apply(params,x)
    r = jnp.mean((p-y)**2)
    return r


class CNN(nn.Module):
  channels: int
  down_scale: int
  num_convs: int
  conv_len: int
  dense_size: int
  num_dense: int
  predictions: int
  features_per_prediction: int
  batch_norm: bool
  dropout: float
  nonconv_features: int
  padding: str

  def setup(self):
    pass

  # TODO: Implement array of convolutions
  # TODO: Implement array of dense
  @nn.compact
  def __call__(self, x, train: bool):
    if self.nonconv_features > 0:
      last_features = x[:,-1,-self.nonconv_features:]
      rest = x[:,:,:-self.nonconv_features]

      x = rest
    # Convolutions.
    for i in range(0,self.num_convs):
      x = nn.Conv(features=self.channels, kernel_size=(self.conv_len,), padding=self.padding)(x)

      if self.batch_norm:
        x = nn.BatchNorm(use_running_average=not train)(x)

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(self.down_scale,), strides=(self.down_scale,))

    x = x.reshape((x.shape[0], -1))  # flatten


    if self.nonconv_features > 0:
      x = jnp.concatenate([x, last_features], axis=1)
    # Dense layers.
    for i in range(0,self.num_dense):
      x = nn.Dense(features=self.dense_size, kernel_init=initializers.glorot_uniform())(x)

      if self.batch_norm:
        x = nn.BatchNorm(use_running_average=not train)(x)

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

      x = nn.relu(x)

    x = nn.Dense(features=self.predictions, kernel_init=initializers.glorot_uniform())(x)
    x = nn.Dense(features=self.predictions*self.features_per_prediction, kernel_init=initializers.glorot_uniform())(x)
    x = nn.sigmoid(x)
    x = x.reshape((-1, self.predictions, self.features_per_prediction))
