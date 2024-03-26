from math import sin, cos, pi
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import numpy as np
import jax
from jax import numpy as jnp
from jax import grad

# USE SIN(THETA), COS(THETA) in Q FUNCTION

# Static
const_l = 3
mu_c = 3.0
mu_p = 2.0
const_g = 9.81
m_pole = 1.0
m_cart = 3.0
eps = 0.01
MAX_FORCE = 1.0
MAX_STEP=2000

ACTIONS = np.array([-MAX_FORCE, MAX_FORCE]) #arange(-MAX_FORCE, MAX_FORCE)

INDEX_X = 0
INDEX_V = 1
INDEX_THETA = 2
INDEX_THETA_DOT = 3
INDEX_SIN_THETA = 4
INDEX_COS_THETA = 5


# State:
class PoleCartState:

  def __init__(self, x, v, theta, theta_dot):
    self.vec = np.zeros(6)
    self.vec[INDEX_X] = x
    self.vec[INDEX_V] = v
    self.vec[INDEX_THETA] = theta % (2 * pi)
    self.vec[INDEX_THETA_DOT] = theta_dot

    self.vec[INDEX_SIN_THETA] = sin(theta)
    self.vec[INDEX_COS_THETA] = cos(theta)

  def __str__(self):
    return (f"theta: {self.vec[INDEX_THETA]:.3f} theta': {self.vec[INDEX_THETA_DOT]:.3f}")


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
  return (cos(state_new.vec[INDEX_THETA]) - cos(state.vec[INDEX_THETA]))

@jax.jit
# Returns ACTION.shape[0] many outputs. Q value for each action.
def q_function(state_vec, params):
  layer1 = jax.nn.relu(state_vec @ params[0] + params[1])
  layer2 = jax.nn.relu(layer1 @ params[2] + params[3])
  result = 10 * jnp.tanh(layer2 @ params[4])
  return result


# Returns the action index with the best Q value for the given state.
def q_policy(state, params, explore=False):
  # Simple linear combination of the state params for Q.

  values = q_function(state_vec=state.vec, params=params)
  action_index = np.argmax(values)

  if explore and random.random() < 0.1:
    action_index = random.randint(0,ACTIONS.shape[0]-1)

  #print("q_policy:", state, values)

  return action_index, values[0][action_index]


def q_policy_noval(state, params, explore=False):
  return q_policy(state, params, explore=explore)[0]


def state_derivative(state, force):
  sin_theta = sin(state.vec[INDEX_THETA])
  cos_theta = cos(state.vec[INDEX_THETA])

  a = m_pole * const_g * sin_theta * cos_theta
  a -= 7 / 3 * (force + m_pole * const_l * state.vec[INDEX_THETA_DOT]**2 * sin_theta -
                mu_c * state.vec[INDEX_V])
  a -= mu_p * state.vec[INDEX_THETA_DOT] * cos_theta / const_l
  a /= m_pole * cos_theta * cos_theta - 7 / 3 * (m_pole + m_cart)
  theta_dd = 3 / (7 * const_l) * (const_g * sin_theta - a * cos_theta -
                            mu_p * state.vec[INDEX_THETA_DOT] / (m_pole * const_l))

  return np.array([state.vec[INDEX_THETA_DOT], theta_dd, state.vec[INDEX_V], a])


def time_step(state, force):
  _, theta_dd, _, a = state_derivative(state, force=force)
  theta = state.vec[INDEX_THETA] + state.vec[
      INDEX_THETA_DOT] * eps + 1 / 2 * theta_dd * eps**2
  theta_d = state.vec[INDEX_THETA_DOT] + theta_dd * eps
  x = state.vec[INDEX_X] + state.vec[INDEX_V] * eps + 1 / 2 * a * eps**2
  v = state.vec[INDEX_V] + a * eps

  return PoleCartState(x=x, v=v, theta=theta, theta_dot=theta_d)


# REMOVE action_idx - use all
# Q^{*}(s,a) = \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma \max\limits_{a'} Q^{*}(s',a')\right]
def improved_q_value(state, action_idx, state_new, params):
  gamma = 0.95
  _, best_next_value = q_policy(state_new, params, explore=False)
  improved_current_value = reward(state, action_idx, state_new) + gamma * best_next_value
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

  draw.line(
      (xx(state.vec[INDEX_X]), yy(1), xx(state.vec[INDEX_X] + force), yy(1)),
      fill=(255, 0, 0, 128))

  draw.line((xx(x_end), yy(y_end), xx(state.vec[INDEX_X]), yy(0)),
            fill=(255, 255, 255, 128))

  return im


def evaluate(start_state, policy, params, image_fn=None):
  step = 0
  images = []
  state = start_state

  state_vecs = []
  action_idxs = []
  improved_q_vec = []

  while True:
    step += 1

    action_index = policy(state, params=params, explore=True)
    state_new = time_step(state, force=ACTIONS[action_index])
    state_vecs.append(state.vec)

    action_idxs.append(action_index)
    q_new = improved_q_value(state, action_index, state_new, params=params)
    if abs(state.vec[INDEX_THETA] - pi) < 0.75 * pi:
      q_new = -1.0 * np.ones(ACTIONS.shape[0])

    improved_q_vec.append(q_new)

    if step % 25 == 0 and image_fn:
      images.append(draw(step, state, force=ACTIONS[action_index]))
    if abs(state.vec[INDEX_THETA] - pi) < 0.75 * pi or step > MAX_STEP:
      break
    state = state_new

  if image_fn:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return step, state_vecs, action_idxs, improved_q_vec


key = jax.random.key(0)

def run_episodes(iteration, num_episodes, params):
  step_avg = 0
  s_vecs = []
  a_idxs = []
  q_vecs = []
  print(i, "run:")
  for cur_episode in tqdm(range(0, num_episodes)):
    start_state = PoleCartState(x=0.0, v=random.gauss() * 0.01, theta=random.gauss() * 0.01 * pi, theta_dot=random.gauss() * 0.05)
    episode_steps, state_vecs, action_idxs, q_v = evaluate(start_state,
                        policy=q_policy_noval,
                        params=params,
                        image_fn=None)#f"q_policy{iteration}.{cur_episode}.gif")
    s_vecs.extend(state_vecs)
    a_idxs.extend(action_idxs)
    q_vecs.extend(q_v)
    step_avg += episode_steps

  print(i, "avg:", step_avg / num_episodes)

  s_vecs = np.stack(s_vecs, axis=0)
  a_idxs = np.stack(a_idxs, axis=0)
  q_vecs = np.stack(q_vecs, axis=0)
  return s_vecs, a_idxs, q_vecs


params = [0.2 * (np.random.random([6, 25]) - 0.5), 
          0.2 * (np.random.random([1,25])-0.5),
          0.2 * (np.random.random([25, 25]) - 0.5), 
          0.2 * (np.random.random([1,25])-0.5),
          0.2*np.random.random([25,ACTIONS.shape[0]])]

def optimize_model(s_vecs, a_idxs, q_vecs, params):
  print("Optimizing Model")
  print("s_vecs:", s_vecs.shape)
  print("a_idxs:", a_idxs.shape)
  print("q_vecs:", q_vecs.shape)

  def loss(params):
    # TODO:::::::::::::::: only use the action_index
    q_preds = q_function(s_vecs, params) # shape(batch, action)
    #print("loss", q_preds.shape, q_vecs.shape)
    #print("loss", q_preds[0], q_vecs[0])
    loss = jnp.average((q_vecs - q_preds)**2)
    return loss

  it = 0
  l_old = None
  params_new = params
  while True:
    g = grad(loss)(params_new)
    l = loss(params_new)
    if it % 100 == 0:
      print(f"{it}. loss=", l, l_old)
      if l_old and l > 0.99 * l_old:
        break
      l_old = l
    it += 1
    for k in range(len(params_new)):
      params_new[k] = params_new[k] - 0.01 * g[k]
  return params_new

for i in range(0, 100):
  s_vecs, a_idxs, q_vecs = run_episodes(i, num_episodes=100, params=params)
  print("Optimizing Q")
  params = optimize_model(s_vecs, a_idxs, q_vecs, params)
