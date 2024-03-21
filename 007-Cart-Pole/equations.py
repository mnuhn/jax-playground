from math import sin, cos, pi
from PIL import Image, ImageDraw
import random
import numpy as np
import jax
from jax import numpy as jnp
from jax import grad

# USE SIN(THETA), COS(THETA) in Q FUNCTION

# Static
l = 3
mu_c = 3.0
mu_p = 2.0
const_g = 9.81
m_pole = 1.0
m_cart = 3.0
eps = 0.01
MAX_FORCE = 10.0

INDEX_X = 0
INDEX_V = 1
INDEX_THETA = 2
INDEX_THETA_DOT = 3


# State:
class PoleCartState:

  def __init__(self, x, v, theta, theta_dot):
    self.vec = np.zeros(4)
    self.vec[INDEX_X] = x
    self.vec[INDEX_V] = v
    self.vec[INDEX_THETA] = theta % (2 * pi)
    self.vec[INDEX_THETA_DOT] = theta_dot

  def __str__(self):
    return (f"vec: {self.vec}")


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
def reward(state, action):
  state_new = time_step(state, force=action)
  return cos(state_new.vec[INDEX_THETA]) - abs(
      state_new.vec[INDEX_THETA_DOT])  # - cos(state.vec[INDEX_THETA])


@jax.jit
def q_function(state_action_vec, params):
  layer1 = jax.nn.sigmoid(state_action_vec @ params)
  result = jnp.average(layer1, axis=1)
  return result


# Returns the action (=force) with the best Q value for the given state.
def q_policy(state, params, explore=False):
  # Simple linear combination of the state params for Q.

  best_force = 0
  best_value = None

  # To simplify, just use a few discrete options for the force.
  tries = np.arange(-MAX_FORCE, MAX_FORCE)
  random.shuffle(tries)
  for force in tries:
    state_action_vec = jnp.expand_dims(np.append(state.vec, force), axis=0)
    value = q_function(state_action_vec, params)[0]
    if best_value is None or value > best_value:
      best_value = value
      best_force = force
    if explore and random.random() < 0.05:
      # simple randomness
      return force, value

  return best_force, best_value


def q_policy_noval(state, params, explore=False):
  return q_policy(state, params, explore=explore)[0]


def state_derivative(state, force):
  sin_theta = sin(state.vec[INDEX_THETA])
  cos_theta = cos(state.vec[INDEX_THETA])

  a = m_pole * const_g * sin_theta * cos_theta
  a -= 7 / 3 * (force + m_pole * l * state.vec[INDEX_THETA_DOT]**2 * sin_theta -
                mu_c * state.vec[INDEX_V])
  a -= mu_p * state.vec[INDEX_THETA_DOT] * cos_theta / l
  a /= m_pole * cos_theta * cos_theta - 7 / 3 * (m_pole + m_cart)
  theta_dd = 3 / (7 * l) * (const_g * sin_theta - a * cos_theta -
                            mu_p * state.vec[INDEX_THETA_DOT] / (m_pole * l))

  return np.array([state.vec[INDEX_THETA_DOT], theta_dd, state.vec[INDEX_V], a])


def time_step(state, force):
  _, theta_dd, _, a = state_derivative(state, force=force)
  theta = state.vec[INDEX_THETA] + state.vec[
      INDEX_THETA_DOT] * eps + 1 / 2 * theta_dd * eps**2
  theta_d = state.vec[INDEX_THETA_DOT] + theta_dd * eps
  x = state.vec[INDEX_X] + state.vec[INDEX_V] * eps + 1 / 2 * a * eps**2
  v = state.vec[INDEX_V] + a * eps

  return PoleCartState(x=x, v=v, theta=theta, theta_dot=theta_d)


def improved_q_value(state, action, params):
  gamma = 0.3
  best_next_value = q_policy(state, params)[1]
  improved_current_value = reward(state, action) + gamma * best_next_value
  return improved_current_value


def draw(step, state, force):
  x_end = state.vec[INDEX_X] + l * sin(state.vec[INDEX_THETA])
  y_end = -l * cos(state.vec[INDEX_THETA])

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

  state_action_vecs = []
  improved_q_vec = []

  while True:
    step += 1

    force = policy(state, params=params, explore=True)
    force = max(min(force, MAX_FORCE), -MAX_FORCE)

    state = time_step(state, force=force)
    state_action_vec = np.append(state.vec, [force])

    state_action_vecs.append(state_action_vec)
    q_new = improved_q_value(state, action=force, params=params)
    improved_q_vec.append([q_new])

    if step % 25 == 0 and image_fn:
      images.append(draw(step, state, force))
    if abs(state.vec[INDEX_THETA] - pi) < 0.75 * pi:
      break

  if image_fn:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return state_action_vecs, improved_q_vec


key = jax.random.key(0)

params = 0.2 * (np.random.random([5, 5]) - 0.5)

for i in range(0, 100):
  a_vecs = []
  q_vecs = []
  for j in range(0, 10):
    print(f"{i} {j}")
    start_state = PoleCartState(x=0.0, v=0.0, theta=0.01 * pi, theta_dot=0.0)
    print(params)
    a_v, q_v = evaluate(start_state,
                        policy=q_policy_noval,
                        params=params,
                        image_fn=f"q_policy{i}.{j}.gif")
    for a in a_v:
      a_vecs.append(a)
    for q in q_v:
      q_vecs.append(q)

  print("Optimizing Q")
  a_vecs = np.stack(a_vecs, axis=0)
  q_vecs = np.stack(q_vecs, axis=0)

  predicted_qs = q_function(a_vecs, params)
  predicted_qs = np.expand_dims(predicted_qs, axis=1)

  def loss(params):
    q_preds = q_function(a_vecs, params)
    loss = jnp.average((q_vecs - q_preds)**2)
    return loss

  for i in range(100):
    g = grad(loss)(params)
    if i % 25 == 0:
      print(f"{i}. loss=", loss(params))
    params = params - 0.01 * g

#start_state = PoleCartState(x=0.0, v=0.0, theta=0.01 * pi, theta_dot=0.5)
#evaluate(start_state, policy=q_policy_noval, image_fn="q_policy.gif")
