import flax.linen as nn
import jax.numpy as jnp

class CNN(nn.Module):
  channels: int
  down_scale: int
  conv_len: int
  dense_size: int
  predictions: int

  def setup(self):
    pass

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.channels, kernel_size=(1, self.conv_len))(x)
    x = nn.sigmoid(x)
    x = nn.max_pool(x, window_shape=(1, self.down_scale), strides=(1, self.down_scale))
    x = nn.Conv(features=self.channels, kernel_size=(1, self.conv_len))(x)
    x = nn.sigmoid(x)
    x = nn.max_pool(x, window_shape=(1, self.down_scale), strides=(1, self.down_scale))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=self.dense_size)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(features=self.predictions)(x)
    x = nn.sigmoid(x)
    return x

  def loss(self, params, x, y):
    p = self.apply(params,x)
    r = jnp.mean((p-y)**2)
    return r

def loss_pred(p,y):
  return jnp.mean((p-y)**2)
