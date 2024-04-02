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


def random_params():
  dims = [5, 100, 100, 100, state.ACTIONS.shape[0]]
  params = []
  for i in range(0, len(dims) - 1):
    params.append(
        np.random.normal(size=[dims[i], dims[i + 1]], scale=(1 / dims[i])**0.5))
    params.append(np.zeros([1, dims[i + 1]]))

  return params


@jax.jit
# Returns ACTION.shape[0] many outputs. Q value for each action.
def q_function(state_vecs, params):
  #jax.debug.print("{state_vecs.shape}", state_vecs=state_vecs)
  layer1 = jax.nn.relu(state_vecs[:, 1:] @ params[0] + params[1])
  layer2 = jax.nn.relu(layer1 @ params[2] + params[3])
  layer3 = jax.nn.relu(layer2 @ params[4] + params[5])
  result = jax.nn.tanh(layer3 @ params[6] + params[7])
  #jax.debug.print("{result.shape}", result=result)
  return result


# Returns the action index with the best Q value for the given state.
def q_policy(cur_state, params, explore_prob=0.0):
  # Simple linear combination of the state params for Q.

  values = q_function(state_vecs=jnp.expand_dims(cur_state.vec, axis=0),
                      params=params)
  #if state.step <= 1:
  #  print(values)
  action_index = np.argmax(values)

  if random.random() < explore_prob:
    action_index = random.randint(0, state.ACTIONS.shape[0] - 1)
  #print("q_policy:", state, values)

  return action_index, values[0][action_index]


def q_policy_noval(cur_state, params, explore_prob=0.0):
  return q_policy(cur_state, params, explore_prob=explore_prob)[0]


# REMOVE action_idx - use all
# Q^{*}(s,a) = \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma \max\limits_{a'} Q^{*}(s',a')\right]
def improved_q_value(cur_state, action_idx, state_new, gamma, params):
  r = state.reward(cur_state, action_idx, state_new)
  improved_current_value = r
  if gamma > 0.0:
    _, best_next_value = q_policy(state_new, params, explore_prob=0.0)

    if abs(state.vec[INDEX_THETA] - pi) < 0.5 * pi:
      best_next_value = 0.0

    improved_current_value += gamma * best_next_value

  #if state.step == 0:
  #  print("improved_current_value", action_idx, "reward", r, "value", improved_current_value)
  return improved_current_value


def optimize_model(epoch, s_vecs, a_idxs, q_vecs, params=None):
  if params == None:
    params = random_params()
  summary_writer = tensorboard.SummaryWriter(f"./tensorboard/{epoch}")
  print("Optimizing Model")
  print("s_vecs:", s_vecs.shape)
  print("a_idxs:", a_idxs.shape)
  print("q_vecs:", q_vecs.shape)

  TEST_SPLIT = 2000
  assert len(s_vecs) > 2 * TEST_SPLIT
  test_S = s_vecs[:TEST_SPLIT]
  test_A = a_idxs[:TEST_SPLIT]
  test_Q = q_vecs[:TEST_SPLIT]
  train_S = s_vecs[TEST_SPLIT:]
  train_A = a_idxs[TEST_SPLIT:]
  train_Q = q_vecs[TEST_SPLIT:]

  def loss(S, A, Q, params):
    Q_preds = q_function(S, params)  # shape(batch, action)
    rows = np.arange(Q_preds.shape[0])
    Q_preds = Q_preds[rows, A]
    #jax.debug.print("Q={Q.shape}, Q_preds={Q_preds.shape}", Q=Q, Q_preds=Q_preds)
    #jax.debug.print("Q={Q}, Q_preds={Q_preds}", Q=Q, Q_preds=Q_preds)
    #loss = jaxopt.loss.huber_loss(Q, Q_preds, delta=1.0)
    #jax.debug.print("loss={loss}", loss=loss)
    #loss = jnp.average(loss)
    loss = jnp.average((Q - Q_preds)**2)**0.5
    return loss

  batch_size = 512
  tx = optax.adam(learning_rate=0.005)
  opt_state = tx.init(params)
  patience = 3 * int(len(train_S) / batch_size)
  print(f"patience: {patience}")
  early_stop = EarlyStopping(min_delta=0.00001, patience=patience)

  it = 0
  params_new = params
  params_out = params

  def getbatch(S, A, Q, batch_size):
    while True:
      perm = np.random.permutation(S.shape[0])
      S = S[perm]
      A = A[perm]
      Q = Q[perm]
      for i in range(0, int(len(S) / batch_size)):
        yield S[i * batch_size:(i + 1) *
                batch_size, :], A[i * batch_size:(i + 1) *
                                  batch_size], Q[i * batch_size:(i + 1) *
                                                 batch_size]

  batcher = getbatch(train_S, train_A, train_Q, batch_size)

  pbar = tqdm()

  while True:
    pbar.update()
    batch_S, batch_A, batch_Q = next(batcher)
    g_params = grad(loss, argnums=3)(batch_S, batch_A, batch_Q, params_new)
    train_l = loss(batch_S, batch_A, batch_Q, params_new)
    test_l = loss(test_S, test_A, test_Q, params_new)

    if it % 25 == 0:
      pbar.set_description(f"train_loss={train_l:.3f} test_loss={test_l:.3f}")
      summary_writer.scalar('test_loss', test_l, it)
      summary_writer.scalar('train_loss', train_l, it)

    early_stop = early_stop.update(test_l)

    if early_stop.has_improved:
      params_out = params_new

    if early_stop.should_stop:
      break

    updates, opt_state = tx.update(g_params, opt_state)
    params_new = optax.apply_updates(params_new, updates)
    it += 1
  del pbar

  return params_out
