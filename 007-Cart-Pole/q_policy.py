import jax
import optax
from tqdm import tqdm
from jax import grad
from flax.training.early_stopping import EarlyStopping
from flax.metrics import tensorboard
from jax import numpy as jnp
import numpy as np
import state
from math import sin, cos, pi
import random

key = jax.random.key(0)


# Discrete Q table: Shape [theta_dot, theta, action].
def random_params(dim):
  # theta_dot, theta, action
  params = np.zeros(shape=[1, dim, state.ACTIONS.shape[0]])
  #params = np.random.normal(size=[dim,dim,state.ACTIONS.shape[0]], scale=0.01)
  return params


def idxs(state_vecs, params):
  assert len(state_vecs.shape) == 2
  assert state_vecs.shape[1] == 6
  MAX_SPEED = 0.1
  theta_dot_idxs = jnp.floor(
      params.shape[0] * (MAX_SPEED + state_vecs[:, state.INDEX_THETA_DOT]) /
      (2 * MAX_SPEED)).astype(int)
  theta_idxs = jnp.floor(params.shape[1] * state_vecs[:, state.INDEX_THETA] /
                         (2 * jnp.pi)).astype(int)

  # Clamp the indices to ensure they are within valid ranges
  theta_dot_idxs = jnp.clip(theta_dot_idxs, 0, params.shape[0] - 1)
  theta_idxs = jnp.clip(theta_idxs, 0, params.shape[1] - 1)

  res = jnp.column_stack([theta_dot_idxs, theta_idxs])
  return res


@jax.jit
# Returns ACTION.shape[0] many outputs. Q value for each action.
def q_function(state_vecs, params):
  assert len(state_vecs.shape) == 2
  assert state_vecs.shape[1] == 6
  idx = idxs(state_vecs, params)
  return params[idx[:, 0], idx[:, 1], :]


# Returns the action index with the best Q value for the given state.
def q_policy(cur_state, params, explore_prob=0.0):
  values = q_function(state_vecs=jnp.expand_dims(cur_state.vec, axis=0),
                      params=params)
  assert len(values.shape) == 2
  assert values.shape[0] == 1
  assert values.shape[1] == 2

  # Randomize if multiple values are best.
  max_val = np.max(values[0])
  max_idxs = np.ravel(np.where(values[0] == max_val))
  action_index = np.random.choice(max_idxs)

  #if cur_state.step <= 1:
  #print(f"Q_function for state={cur_state}: {values}")
  #print(f"Action: {action_index}")

  if random.random() < explore_prob:
    action_index = random.randint(0, state.ACTIONS.shape[0] - 1)
  #print("q_policy:", state, values)

  #if cur_state.step < 5 and explore_prob == 0:
  #  print("qfunction", cur_state, values, action_index)

  return action_index, values[0][action_index]


def improved_q_value(cur_state, action_idx, state_new, gamma, params):
  r = cur_state.reward(action_idx, state_new)
  improved_current_value = r
  if gamma > 0.0:
    if r == 0.0:
      return 0.0
    _, best_next_value = q_policy(state_new, params, explore_prob=0.0)
    improved_current_value += gamma * best_next_value
    #print("improved:", r, "+", best_next_value)

  #if state.step == 0:
  #  print("improved_current_value", action_idx, "reward", r, "value", improved_current_value)
  return improved_current_value


# This is far from the optimal implementation, but it is compatible/swappable with a NN policy.
def optimize_model(epoch, s_vecs, a_idxs, q_vecs, alpha, params):
  assert len(a_idxs.shape) == 1
  assert len(s_vecs.shape) == 2
  assert len(q_vecs.shape) == 2
  assert q_vecs.shape[1] == 1

  s_idxs = idxs(s_vecs, params)

  for s_idx in tqdm(jnp.unique(s_idxs, axis=0), desc="Optimize Q"):
    s_mask = (s_idxs == s_idx).all(axis=1)
    for a_idx in jnp.unique(a_idxs[s_mask]):
      a_mask = (a_idxs == a_idx) & s_mask
      #print("idx", s_idx, a_idx, jnp.average(q_vecs[a_mask]))
      #print(a_mask)
      #print(np.argwhere(a_mask==True))
      #print(s_vecs[a_mask])
      #print(q_vecs[a_mask])
      #params[s_idx[0], s_idx[1], a_idx] =
      #print()
      params[s_idx[0], s_idx[1],
             a_idx] = (1 - alpha) * (params[s_idx[0], s_idx[1], a_idx]
                                    ) + alpha * jnp.average(q_vecs[a_mask])
  #params[i[0],i[1],a_idxs] = 0.5 * params[i[0],i[1],a_idxs] + 0.5* q_vecs[:,0]

  with np.printoptions(precision=3, suppress=True, threshold=np.inf):
    print(params[:, :, 0])
    print(params[:, :, 1])
  print()

  return params
