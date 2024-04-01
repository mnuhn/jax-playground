from math import sin, cos, pi
from PIL import Image, ImageDraw
from tqdm import tqdm
from flax.metrics import tensorboard
import random
import numpy as np
import jax
import jaxopt
from jax import numpy as jnp
from jax import grad
import optax
from flax.training.early_stopping import EarlyStopping

# Static
const_l = 3
mu_c = 3.0
mu_p = 2.0
const_g = 9.81
m_pole = 1.0
m_cart = 3.0
eps = 0.01
MAX_FORCE = 25.0
MAX_STEP = 2000

MAX_STEP_KEEP = 25000
MAX_STEP_MORE = 25000

NUM_EPISODES = 100
NUM_EPOCHS = 50

ACTIONS = np.array([-MAX_FORCE, MAX_FORCE])  #arange(-MAX_FORCE, MAX_FORCE)

INDEX_X = 0
INDEX_V = 1
INDEX_THETA = 2
INDEX_THETA_DOT = 3
INDEX_SIN_THETA = 4
INDEX_COS_THETA = 5


# State:
class PoleCartState:

  def __init__(self, x, v, theta, theta_dot, step=0):
    self.vec = np.zeros(6)
    self.vec[INDEX_X] = x
    self.vec[INDEX_V] = v
    self.vec[INDEX_THETA] = theta % (2 * pi)
    self.vec[INDEX_THETA_DOT] = theta_dot

    self.vec[INDEX_SIN_THETA] = sin(theta)
    self.vec[INDEX_COS_THETA] = cos(theta)

    self.step = step

  def __str__(self):
    return (
        f"theta: {self.vec[INDEX_THETA]:.3f} theta': {self.vec[INDEX_THETA_DOT]:.3f}"
    )


# Action: $a \in [-MAX_FORCE, MAX_FORCE]$


# All policies $\pi(s)$ are deterministic.
def move_nothing(state, params=None):
  return 0.0


def move_constant(state, params=None):
  return 3.0


def move_random(state, params=None):
  return 10 * (random.random() - 0.5)


def move_opposite(state, params=None):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.vec[INDEX_THETA] - pi)
  return -angle * 10


def move_opposite_upswing(state, params=None):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.vec[INDEX_THETA] - pi)
  if abs(angle) > pi * 0.5:
    return -angle * 10
  return angle * 10


# TODO: IMPLEMENT Q LEARNING - WORK IN PROGRESS


# Use the delta in y component above zero as reward.
def reward(state, action_index, state_new):
  return cos(state_new.vec[INDEX_THETA]) - 0.2 * state_new.vec[INDEX_THETA_DOT]


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
def q_policy(state, params, explore_prob=0.0):
  # Simple linear combination of the state params for Q.

  values = q_function(state_vecs=jnp.expand_dims(state.vec, axis=0),
                      params=params)
  #if state.step <= 1:
  #  print(values)
  action_index = np.argmax(values)

  if random.random() < explore_prob:
    action_index = random.randint(0, ACTIONS.shape[0] - 1)
  #print("q_policy:", state, values)

  return action_index, values[0][action_index]


def q_policy_noval(state, params, explore_prob=0.0):
  return q_policy(state, params, explore_prob=explore_prob)[0]


def state_derivative(state, force):
  sin_theta = sin(state.vec[INDEX_THETA])
  cos_theta = cos(state.vec[INDEX_THETA])

  a = m_pole * const_g * sin_theta * cos_theta
  a -= 7 / 3 * (force + m_pole * const_l * state.vec[INDEX_THETA_DOT]**2 *
                sin_theta - mu_c * state.vec[INDEX_V])
  a -= mu_p * state.vec[INDEX_THETA_DOT] * cos_theta / const_l
  a /= m_pole * cos_theta * cos_theta - 7 / 3 * (m_pole + m_cart)
  theta_dd = 3 / (7 * const_l) * (const_g * sin_theta - a * cos_theta -
                                  mu_p * state.vec[INDEX_THETA_DOT] /
                                  (m_pole * const_l))

  return np.array([state.vec[INDEX_THETA_DOT], theta_dd, state.vec[INDEX_V], a])


def time_step(state, force):
  _, theta_dd, _, a = state_derivative(state, force=force)
  theta = state.vec[INDEX_THETA] + state.vec[
      INDEX_THETA_DOT] * eps + 1 / 2 * theta_dd * eps**2
  theta_d = state.vec[INDEX_THETA_DOT] + theta_dd * eps
  x = state.vec[INDEX_X] + state.vec[INDEX_V] * eps + 1 / 2 * a * eps**2
  v = state.vec[INDEX_V] + a * eps

  return PoleCartState(x=x,
                       v=v,
                       theta=theta,
                       theta_dot=theta_d,
                       step=state.step + 1)


# REMOVE action_idx - use all
# Q^{*}(s,a) = \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma \max\limits_{a'} Q^{*}(s',a')\right]
def improved_q_value(state, action_idx, state_new, params):
  gamma = 0.0

  r = reward(state, action_idx, state_new)
  improved_current_value = r
  if gamma > 0.0:
    _, best_next_value = q_policy(state_new, params, explore_prob=0.0)
    improved_current_value += gamma * best_next_value

  #if state.step == 0:
  #  print("improved_current_value", action_idx, "reward", r, "value", improved_current_value)
  return improved_current_value


def draw(step, state, force):
  x_end = state.vec[INDEX_X] + const_l * sin(state.vec[INDEX_THETA])
  y_end = -const_l * cos(state.vec[INDEX_THETA])

  def xx(x_in):
    return 200 + int(x_in * 10)

  def yy(y_in):
    return 50 + int(y_in * 10)

  im = Image.new(mode="RGB", size=(400, 100))
  draw = ImageDraw.Draw(im)

  draw.text((0, 0), f"step={step: >5d}")

  draw.text((0, 70), f"f ={force: >+5.1f}N", fill=(255, 0, 0, 255))
  draw.text((0, 80), f"x ={state.vec[INDEX_X]: >+5.1f}m")
  draw.text((0, 90), f"x'={state.vec[INDEX_V]: >+5.1f}m/s")

  draw.text((80, 80), f"t ={state.vec[INDEX_THETA]/pi*180: >+6.1f}°")
  draw.text((80, 90), f"t'={state.vec[INDEX_THETA_DOT]/pi*180: >+6.1f}°/s")

  draw.line(
      (xx(state.vec[INDEX_X] - 1), yy(0), xx(state.vec[INDEX_X] + 1), yy(0)),
      fill=(255, 255, 255, 128))

  draw.line((xx(
      state.vec[INDEX_X]), yy(1), xx(state.vec[INDEX_X] + force / 10.0), yy(1)),
            fill=(255, 0, 0, 128))

  draw.line((xx(x_end), yy(y_end), xx(state.vec[INDEX_X]), yy(0)),
            fill=(255, 255, 255, 128))

  return im


def evaluate(start_state, policy, params, image_fn=None, explore_prob=0.0):
  step = 0
  images = []
  state = start_state

  states = []
  new_states = []
  action_idxs = []
  improved_q_vec = []

  while True:
    step += 1

    action_index = policy(state, params=params, explore_prob=explore_prob)
    state_new = time_step(state, force=ACTIONS[action_index])

    states.append(state)
    new_states.append(state_new)
    action_idxs.append(action_index)

    if step % 25 == 0 and image_fn:
      images.append(draw(step, state, force=ACTIONS[action_index]))
    if abs(state.vec[INDEX_THETA] - pi) < 0.5 * pi or step > MAX_STEP:
      break
    state = state_new

  if image_fn and len(images) > 0:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return step, states, action_idxs, new_states


key = jax.random.key(0)


def run_episodes(iteration, num_episodes, params, explore_prob):
  step_avg = 0
  states = []
  new_states = []
  a_idxs = []
  q_vecs = []
  print(i, "run:")
  for cur_episode in tqdm(range(0, num_episodes)):
    start_state = PoleCartState(
        x=0.0,
        v=0.0,  #random.gauss() * 0.1,
        theta=0.1 + random.gauss() * 0.05,
        theta_dot=random.gauss() * 0.001)
    episode_steps, cur_states, action_idxs, cur_new_states = evaluate(
        start_state,
        policy=q_policy_noval,
        params=params,
        image_fn=f"q_policy{iteration}.{cur_episode}.gif",
        explore_prob=explore_prob)

    states.extend(cur_states)
    a_idxs.extend(action_idxs)
    new_states.extend(cur_new_states)

    step_avg += episode_steps

  print(i, "avg:", step_avg / num_episodes)

  return states, a_idxs, new_states


def random_params():
  dims = [5, 50, 50, 50, ACTIONS.shape[0]]
  params = []
  for i in range(0, len(dims) - 1):
    params.append(
        np.random.normal(size=[dims[i], dims[i + 1]], scale=(1 / dims[i])**0.5))
    params.append(np.zeros([1, dims[i + 1]]))

  return params


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

  batch_size = 256
  tx = optax.adam(learning_rate=0.005)
  opt_state = tx.init(params)
  patience = int(len(train_S) / batch_size)
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


start_state = PoleCartState(x=0.0, v=0.0, theta=0.25 * pi, theta_dot=0.0)
next_state = time_step(start_state, force=10)
print("reward1:", reward(start_state, None, next_state))

next_state = time_step(start_state, force=-10)
print("reward2:", reward(start_state, None, next_state))

s_vecs_all = []
a_idxs_all = []
q_vecs_all = []
explore_prob = 1.0

# TODO: Merge data from multiple episodes.
# Keep early episodes (states)
# Stratified Sampling for Training Data.
all_states = []
all_a_idxs = []
all_new_states = []

params = random_params()

for i in range(0, NUM_EPOCHS):
  print("explore_prob:", explore_prob)
  next_states, next_a_idxs, next_new_states = run_episodes(
      i, num_episodes=NUM_EPISODES, params=params, explore_prob=explore_prob)

  print("shuffle data")
  c = list(zip(all_states, all_a_idxs, all_new_states))
  random.shuffle(c)

  if len(c) > MAX_STEP_KEEP:
    print(f"limit past data ({len(c)}) to {MAX_STEP_KEEP}")
    c = c[:MAX_STEP_KEEP]

  d = list(zip(next_states, next_a_idxs, next_new_states))
  random.shuffle(d)

  if len(d) > MAX_STEP_MORE:
    print(f"limit overall data ({len(d)}) to {MAX_STEP_MORE}")
    d = d[:MAX_STEP_MORE]

  e = c + d

  all_states, all_a_idxs, all_new_states = [list(x) for x in zip(*e)]
  print("done")

  s_vecs = np.stack([state.vec for state in all_states], axis=0)
  a_idxs = np.stack(all_a_idxs, axis=0)
  sn_vecs = np.stack([state.vec for state in all_new_states], axis=0)

  improved_q_vec = []
  for i in range(len(all_states)):
    q_new = improved_q_value(all_states[i],
                             all_a_idxs[i],
                             all_new_states[i],
                             params=params)
    if abs(all_states[i].vec[INDEX_THETA] - pi) < 0.5 * pi:
      q_new = np.array(0.0, dtype=float)
    improved_q_vec.append(q_new)
  improved_q_vec = np.stack(improved_q_vec, axis=0)

  with open(f"./out{i}.txt", "w") as f:
    with np.printoptions(threshold=np.inf):
      f.write(f"S: {s_vecs}\n")
      f.write(f"\n\n\n")
      f.write(f"Q: {improved_q_vec}\n")

  print("Optimizing Q")
  params = optimize_model(i, s_vecs, a_idxs, improved_q_vec, params=params)
  explore_prob *= 0.95
