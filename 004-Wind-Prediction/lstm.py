import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

from flax.linen import initializers
from flax import traverse_util
from jax import tree_util
from visualizer import Visualizer

# Model descriptor looks like this:
# HistoryA:dimA0,dimA1,...;HistoryB:dimB0,dimB1,...
# model_description = "128:10,10;64:5,5"

class LstmDescription:
  history: int
  downsample: int
  layer_dims: list[int]

  def __init__(self, s):
    history_downsample_str, layer_description_str = s.split(":")
    history_str, downsample_str = history_downsample_str.split(",")
    self.history = int(history_str)
    self.downsample = int(downsample_str)
    layer_dims = layer_description_str.split(",")
    self.layer_dims = [ int(x) for x in layer_dims ]

  def __str__(self):
    return f"LSTM \u007b History: {self.history} Dims: {self.layer_dims} \u007d"

class ModelDescription:
  lstms: list[LstmDescription]

  def __init__(self, s):
    self.lstms = []

    for lstm_description_str in s.split(";"):
      lstm = LstmDescription(lstm_description_str)
      self.lstms.append(lstm)

  def __str__(self):
    inner = " ".join([ str(l) for l in self.lstms])
    result = "Model { " + inner + " }"
    return result

class LSTM(nn.Module):
  predictions: int
  model_desc: ModelDescription
  dense_sizes: list[int]
  features_per_prediction: int
  dropout: float
  nonlstm_features: int
  batch_norm: bool

  def setup(self):
    pass

  @nn.compact
  def __call__(self, x, train: bool, debug:bool = False):
    debug_output = {}

    if debug:
      debug_output["input"] = x

    if self.nonlstm_features > 0:
      last_features = x[:,-1,-self.nonlstm_features:]
      rest = x[:,:,:-self.nonlstm_features]

      x = rest

      if debug:
        debug_output["input_lstm"] = rest
        debug_output["last_features"] = last_features

    def lstm_layer(x, dim):
      cell = nn.OptimizedLSTMCell(dim)
      def body_fn(cell, carry, x):
        return cell(carry, x)
      scan = nn.scan(
        body_fn, variable_broadcast="params",
        split_rngs={"params": False}, in_axes=1, out_axes=1)

      input_shape =  x[:, 0, :].shape
      carry = cell.initialize_carry(
        jax.random.key(0), input_shape)
      carry, x = scan(cell, carry, x)

      return x

    lstm_outs = []

    for i, lstm in enumerate(self.model_desc.lstms):
      cur_lstm_x = x[:, -lstm.history:, :]

      if debug:
        debug_output[f"lstm_{i}_in"] = cur_lstm_x

      if lstm.downsample > 1:
        cur_lstm_x = nn.max_pool(cur_lstm_x, window_shape=(lstm.downsample,), strides=(lstm.downsample,))
        if debug:
          debug_output[f"lstm_{i}_downsampled"] = cur_lstm_x

      for j, dim in enumerate(lstm.layer_dims):
        print(i, j, dim)
        cur_lstm_x = lstm_layer(cur_lstm_x, dim)
        if debug:
          debug_output[f"lstm_{i}_{j}"] = cur_lstm_x
        if self.dropout > 0.0:
          cur_lstm_x = nn.Dropout(rate=self.dropout, deterministic=not train)(cur_lstm_x)
      
      cur_lstm_out = cur_lstm_x[:, -1, :]
      cur_lstm_out = cur_lstm_out.reshape((x.shape[0], -1))  # flatten

      if debug:
        debug_output[f"lstm_{i}_last_state"] = cur_lstm_out

      lstm_outs.append(cur_lstm_out)

    x = jnp.concatenate(lstm_outs, axis=1)

    if debug:
      debug_output["lstms_concat"] = x

    if self.nonlstm_features > 0:
      x = jnp.concatenate([x, last_features], axis=1)

    if debug:
      debug_output["lstms_concat_with_nonlstm_features"] = x

    for i in range(0,len(self.dense_sizes)):
      name = f'dense_{i}'
      debug_output[f"{name}_act"] = x

      x = nn.Dense(features=self.dense_sizes[i])(x)

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

      x = nn.relu(x)

      if debug:
        debug_output[f"{name}_relu"] = x

    x = nn.Dense(features=self.predictions*self.features_per_prediction, kernel_init=initializers.glorot_uniform())(x)
    x = x.reshape((-1, self.predictions, self.features_per_prediction))

    if debug:
      debug_output[f"final_act"] = x

    x = nn.sigmoid(x)

    if debug:
      debug_output["final"] = x


    if debug:
      return x, debug_output

    return x

  def loss(self, params, x, y):
    p = self.apply(params,x)
    r = jnp.mean((p-y)**2)
    return r
