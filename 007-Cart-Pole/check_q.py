import q_policy
import state
from jax import numpy as jnp

s1 = state.random_upright_state()
s2 = state.random_upright_state()
s = jnp.stack([s1.vec, s2.vec], axis=0)

p = q_policy.random_params(dim=5)
idxs = q_policy.idxs(s, p)

print("===")
print(idxs)
print("===")
print(p[idxs])
p[idxs[0], idxs[1], 0] = 1.0
print(p)

q_policy.q_function(s, p)

d = q_policy.q_function(s, p)
print("===")
