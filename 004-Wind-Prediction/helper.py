from jax import lax
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

def convolution(x, kernel, div, p):
  r = lax.conv_general_dilated(lhs=x,
                               rhs=kernel,
                               window_strides=(1,1,),
                               padding='VALID', # Use padding.
                               dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                               lhs_dilation=(1,1),
                               rhs_dilation=(1,1))


  # Avg pooling.
  # This tensor product is needed to ensure we keep all 16 channels separate.
  avg_kernel_4d = jnp.einsum('ij,kl',
                              np.array([[1/div]*int(div)]),
                              np.identity(p.channels)) # 1x2x16x16

  # 2nd avg pooling.
  r =  lax.conv_general_dilated(lhs=r,
                                rhs=avg_kernel_4d,
                                window_strides=(1,div),
                                padding='VALID',
                                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                lhs_dilation=(1,1),
                                rhs_dilation=(1,1)) # Bx1xXxHISTORY/2

#  r = lax.reduce_window(
#          operand=r,
#          computation=np.max,
#          init_value=0,
#          window_dimensions=(1,div,),
#          window_strides=(1,div,),
#          padding='VALID',
#          base_dilation=(1,1),
#          window_dilation=(1,1))
  r = jax.nn.sigmoid(r)
  return r

def sin_cos(x, period):
  in_radians = 2.0 * np.pi * x[:,0] / period
  sin = jnp.expand_dims(jnp.sin(month_in_radians), axis=1)
  cos = jnp.expand_dims(jnp.cos(month_in_radians), axis=1)

  return sin, cos
