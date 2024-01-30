import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

from flax.linen import initializers
from flax import traverse_util
from jax import tree_util
from visualizer import Visualizer

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
  conv_channels: list[int]
  down_scale: int
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
  def __call__(self, x, train: bool, debug:bool = False):
    debug_output = {}

    if debug:
      debug_output["input"] = x

    if self.nonconv_features > 0:
      last_features = x[:,-1,-self.nonconv_features:]
      rest = x[:,:,:-self.nonconv_features]

      if debug:
        debug_output["last_features"] = last_features
        debug_output["rest"] = rest
      x = rest

    if debug:
      debug_output["input_conv"] = x

    # Convolutions.
    for i in range(0,len(self.conv_channels)):
      name = f'conv_{i}'
      x = nn.Conv(features=self.conv_channels[i], kernel_size=(self.conv_len,), padding=self.padding)(x)

      if debug:
        debug_output[f"{name}_conv"] = x

      if self.batch_norm:
        x = nn.BatchNorm(use_running_average=not train)(x)

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

      x = nn.relu(x)

      if debug:
        debug_output[f"{name}_relu"] = x

      x = nn.max_pool(x, window_shape=(self.down_scale,), strides=(self.down_scale,))

      if debug:
        debug_output[f"{name}_pooled"] = x

    x = x.reshape((x.shape[0], -1))  # flatten

    if debug:
      debug_output["conv_reshaped"] = x

    if self.nonconv_features > 0:
      x = jnp.concatenate([x, last_features], axis=1)

    if debug:
      debug_output["conv_reshaped_with_nonconv_features"] = x

    # Dense layers.
    for i in range(0,self.num_dense):
      name = f'dense_{i}'
      x = nn.Dense(features=self.dense_size, kernel_init=initializers.glorot_uniform())(x)

      debug_output[f"{name}_act"] = x

      if self.batch_norm:
        x = nn.BatchNorm(use_running_average=not train)(x)

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

      x = nn.relu(x)

      if debug:
        debug_output[f"{name}_relu"] = x

    x = nn.Dense(features=self.predictions*self.features_per_prediction, kernel_init=initializers.glorot_uniform())(x)
    x = x.reshape((-1, self.predictions, self.features_per_prediction))
    debug_output[f"{name}_final_act"] = x

    x = nn.sigmoid(x)
    debug_output[f"{name}_final_sigmoid"] = x

    if debug:
      debug_output["final"] = x

    if debug:
      return x, debug_output

    return x
