import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

from flax.linen import initializers
from flax import traverse_util
from jax import tree_util
from visualizer import Visualizer

# Layers:
# * C: convolutional layer - ch: channels, k: kernel size
# * L: LSTM layer - ch: "channels"
# * D: dense layer - d: dimension
# * M: max pool layer - w: width

def parse_layer_details(char, details_str):
  details = {}
  if char == 'I':
    details['type'] = 'input'
  elif char == 'C':
    details['type'] = 'conv'
  elif char == 'D':
    details['type'] = 'dense'
  elif char == 'L':
    details['type'] = 'lstm'
  elif char == 'M':
    details['type'] = 'maxpool'
  else:
    assert False
  for detail in details_str.split(','):
    key, value = detail.split(':')
    details[key] = int(value)
  return details

def parse_arch(encoded_str):
  stack = []
  current_group = []

  i = 0
  while i < len(encoded_str):
    char = encoded_str[i]

    if char == '[':  # Start of a repeatable group
      stack.append(current_group)
      current_group = []
      i += 1

    elif char in ['I', 'C', 'L', 'M', 'D']:  # Layer indicator
      j = encoded_str.find('}', i + 1)
      layer_details_str = encoded_str[i+2:j]
      layer_details = parse_layer_details(char, layer_details_str)

      current_group.append(layer_details)
      i = j + 1

    elif char == ']':  # End of a repeatable group
      last_group = current_group
      current_group = stack.pop()
      current_group.append(last_group)
      i += 1

    elif char == '|':  # Separator for parallel groups, treat as new group
      current_group = []
      i += 1
    else:
      i += 1

  if stack:
      raise ValueError("Unmatched brackets in the input string.")

  return current_group

class LSTM(nn.Module):
  predictions: int
  model_arch: list
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
      static_features = x[:,-1,-self.nonlstm_features:]
      x = x[:,:,:-self.nonlstm_features]

      if debug:
        debug_output["static_features"] = static_features

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

    def lstm_layer2(x, dim):
      cell = nn.OptimizedLSTMCell(dim)
      #cell = nn.GRUCell(dim)
      cell.initialize_carry(jax.random.key(0), x[:, 0, :].shape)
      rnn = nn.RNN(cell)
      return rnn(x)

    stack_outputs = []
    # CNN/LSTM Layers
    for stack_idx, stack in enumerate(self.model_arch[0]):
      cur_stack_x = x
      cur_stack_out = None
      for piece_idx, piece in enumerate(stack):
        if piece['type'] == 'input':
          assert piece_idx == 0
          f = int(piece['fr'])
          t = int(piece['to'])
          assert f < t
          if t == 0:
            cur_stack_x = cur_stack_x[:, f:, :]
          else:
            cur_stack_x = cur_stack_x[:, f:t, :]
          if debug:
            debug_output[f"stack_{stack_idx}_{piece_idx}_in"] = cur_stack_x
          cur_stack_out = cur_stack_x
        elif piece['type'] == 'conv':
          channels = int(piece['ch'])
          kernel_size = int(piece['k'])
          assert channels > 0
          assert kernel_size > 0
          cur_stack_x = nn.Conv(features=channels, kernel_size=(kernel_size,), padding='SAME')(cur_stack_x)
          if debug:
            debug_output[f"stack_{stack_idx}_{piece_idx}_conv"] = cur_stack_x
          if self.dropout > 0.0:
            cur_stack_x = nn.Dropout(rate=self.dropout, deterministic=not train)(cur_stack_x)
          if self.batch_norm:
            cur_stack_out = nn.BatchNorm(use_running_average=not train)(cur_stack_x) 
          cur_stack_x = nn.relu(cur_stack_x)
          cur_stack_out = cur_stack_x
        elif piece['type'] == 'maxpool':
          width = int(piece['w'])
          assert width > 1
          cur_stack_x = nn.max_pool(cur_stack_x, window_shape=(width,), strides=(width,))
          if debug:
            debug_output[f"stack_{stack_idx}_{piece_idx}_maxpool"] = cur_stack_x
          if self.dropout > 0.0:
            cur_stack_x = nn.Dropout(rate=self.dropout, deterministic=not train)(cur_stack_x)
          if self.batch_norm:
            cur_stack_out = nn.BatchNorm(use_running_average=not train)(cur_stack_x) 
          cur_stack_out = cur_stack_x
        elif piece['type'] == 'lstm':
          dim = int(piece['ch'])
          assert dim > 0
          cur_stack_x = lstm_layer2(cur_stack_x, dim)
          if debug:
            debug_output[f"stack_{stack_idx}_{piece_idx}_lstm"] = cur_stack_x
          if self.dropout > 0.0:
            cur_stack_x = nn.Dropout(rate=self.dropout, deterministic=not train)(cur_stack_x)
          cur_stack_out = cur_stack_x[:, -1, :]
        else:
          assert False
      stack_outputs.append(cur_stack_out)

    assert len(stack_outputs) > 0
    x = jnp.concatenate(stack_outputs, axis=1)
    x = x.reshape((x.shape[0], -1))  # flatten

    if debug:
      debug_output['stacked_outputs'] = x

    x = jnp.concatenate([x, static_features], axis=1)

    if debug:
      debug_output['stacked_outputs_with_static_features'] = x

    ## Dense Layers
    for dense_idx, dense in enumerate(self.model_arch[1]):
      dim = int(dense['d'])
      assert dim > 0

      x = nn.Dense(features=dim)(x)
      debug_output[f'dense_{dense_idx}_out'] = x
      if self.batch_norm:
        x = nn.BatchNorm(use_running_average=not train)(x) 
      x = nn.relu(x)
      debug_output[f'dense_{dense_idx}_relu'] = x

      if self.dropout > 0.0:
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
    
    # Final Layer
    x = nn.Dense(features=self.predictions*self.features_per_prediction, kernel_init=initializers.glorot_uniform())(x)
    x = x.reshape((-1, self.predictions, self.features_per_prediction))
    x = nn.sigmoid(x)

    if debug:
      debug_output['prediction'] = x

    if debug:
      return x, debug_output

    return x


  def loss(self, params, x, y):
    p = self.apply(params,x)
    r = jnp.mean((p-y)**2)
    return r
