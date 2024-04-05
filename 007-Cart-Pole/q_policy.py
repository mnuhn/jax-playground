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


def random_params(dim):
  dims = [5, dim, dim, dim, state.ACTIONS.shape[0]]
  params = []

  for i in range(0, len(dims) - 1):
    params.append(
        np.random.normal(size=[dims[i], dims[i + 1]], scale=(1 / dims[i])))
    params.append(np.zeros([1, dims[i + 1]]))

  return params


@jax.jit
# Returns ACTION.shape[0] many outputs. Q value for each action.
def q_function(state_vecs, params):
  #jax.debug.print("{state_vecs.shape}", state_vecs=state_vecs)

  result = jnp.expand_dims(jnp.absolute(pi - state_vecs[:, state.INDEX_THETA]),
                           axis=1)
  result -= jnp.expand_dims(0.1 * jnp.absolute(state_vecs[:, state.INDEX_X]),
                            axis=1)
  #print(result)

  # Skip X position.
  layer1 = jax.nn.relu(state_vecs[:, 1:] @ params[0] + params[1])
  layer2 = jax.nn.relu(layer1 @ params[2] + params[3])
  layer3 = jax.nn.relu(layer2 @ params[4] + params[5])

  result += jax.nn.tanh(layer3 @ params[6] + params[7])

  #jax.debug.print("{result.shape}: {result}", result=result)
  return result


# Returns the action index with the best Q value for the given state.
def q_policy(cur_state, params, explore_prob=0.0):
  # Simple linear combination of the state params for Q.

  values = q_function(state_vecs=jnp.expand_dims(cur_state.vec, axis=0),
                      params=params)
  action_index = np.argmax(values)

  #if cur_state.step <= 1:
  #  print(f"Q_function for state={cur_state}: {values}")
  #  print(f"Action: {action_index}")

  if random.random() < explore_prob:
    action_index = random.randint(0, state.ACTIONS.shape[0] - 1)
  #print("q_policy:", state, values)

  return action_index, values[0][action_index]


def q_policy_noval(cur_state, params, explore_prob=0.0):
  return q_policy(cur_state, params, explore_prob=explore_prob)[0]


# REMOVE action_idx - use all
# Q^{*}(s,a) = \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma \max\limits_{a'} Q^{*}(s',a')\right]
def improved_q_value(cur_state, action_idx, state_new, gamma, params):
  r = cur_state.reward(action_idx, state_new)
  improved_current_value = r
  if gamma > 0.0:
    _, best_next_value = q_policy(state_new, params, explore_prob=0.0)

    if abs(state_new.vec[state.INDEX_THETA] - pi) < 0.5 * pi:
      best_next_value = 0.0

    improved_current_value += gamma * best_next_value

  #if state.step == 0:
  #  print("improved_current_value", action_idx, "reward", r, "value", improved_current_value)
  return improved_current_value


def optimize_model(epoch,
                   s_vecs,
                   a_idxs,
                   q_vecs,
                   params,
                   learning_rate=0.001,
                   test_split_frac=0.1,
                   min_steps=1000,
                   patience_epochs=5,
                   batch_size=128):
  assert len(a_idxs.shape) == 1
  assert len(s_vecs.shape) == 2
  assert len(q_vecs.shape) == 2
  assert q_vecs.shape[1] == 1

  summary_writer = tensorboard.SummaryWriter(f"./tensorboard/{epoch}")

  TEST_SPLIT = int(test_split_frac * len(s_vecs))
  assert len(s_vecs) > TEST_SPLIT
  test_S = s_vecs[:TEST_SPLIT]
  test_A = a_idxs[:TEST_SPLIT]
  test_Q = q_vecs[:TEST_SPLIT]
  train_S = s_vecs[TEST_SPLIT:]
  train_A = a_idxs[TEST_SPLIT:]
  train_Q = q_vecs[TEST_SPLIT:]

  def loss(S, A, Q, p):
    Q_preds = q_function(S, p)  # shape(batch, action)
    #print("Q1:", Q_preds.shape)
    #print("Q1:", Q_preds)
    rows = np.arange(Q_preds.shape[0])
    #print("Q2:", Q_preds.shape)
    #print("A", A.shape)
    Q_preds = jnp.expand_dims(Q_preds[rows, A], axis=1)
    #print("Q2:", Q_preds)
    #print("Q3:", Q_preds.shape)
    #print("QQ:", Q)
    #jax.debug.print("Q={Q.shape}, Q_preds={Q_preds.shape}", Q=Q, Q_preds=Q_preds)
    #jax.debug.print("Q={Q}, Q_preds={Q_preds}", Q=Q, Q_preds=Q_preds)
    #loss = jaxopt.loss.huber_loss(Q, Q_preds, delta=1.0)
    #loss = jnp.average(loss)
    loss = jnp.average((Q - Q_preds)**2)
    #jax.debug.print("loss={loss}", loss=loss)
    return loss

  params_new = params
  params_out = params

  tx = optax.adam(learning_rate=learning_rate)
  opt_state = tx.init(params_new)
  patience = patience_epochs * int(len(train_S) / batch_size)
  print(f"patience: {patience}")
  early_stop = EarlyStopping(patience=patience)

  it = 0

  def getbatch(S, A, Q, batch_size):
    while True:
      perm = np.random.permutation(S.shape[0])
      S = S[perm]
      A = A[perm]
      Q = Q[perm]
      assert len(S) > batch_size
      for i in range(0, int(len(S) / batch_size)):
        yield S[i * batch_size:(i + 1) *
                batch_size, :], A[i * batch_size:(i + 1) *
                                  batch_size], Q[i * batch_size:(i + 1) *
                                                 batch_size]

  batcher = getbatch(train_S, train_A, train_Q, batch_size)
  pbar = tqdm(total=max(min_steps, early_stop.patience))

  while True:
    pbar.update()
    batch_S, batch_A, batch_Q = next(batcher)
    g_params = grad(loss, argnums=3)(batch_S, batch_A, batch_Q, params_new)
    train_l = loss(batch_S, batch_A, batch_Q, params_new)**0.5
    test_l = loss(test_S, test_A, test_Q, params_new)**0.5

    if it % 25 == 0:
      #print(batch_S)
      #print(batch_Q)
      pbar.set_description(
          f"train_loss={train_l:.8f} test_loss={test_l:.8f} best_test_loss={early_stop.best_metric:.8f}"
      )
      summary_writer.scalar('test_loss', test_l, it)
      summary_writer.scalar('train_loss', train_l, it)

    early_stop = early_stop.update(test_l)

    if it > min_steps and early_stop.has_improved:
      pbar.total = pbar.n + early_stop.patience
      params_out = params_new

    if it > min_steps and early_stop.should_stop:
      break

    updates, opt_state = tx.update(g_params, opt_state)
    params_new = optax.apply_updates(params_new, updates)
    it += 1

  pbar.close()
  print()
  return params_out
