from math import sin, cos, pi
import tqdm
from PIL import Image, ImageDraw
import random

# Static
l = 3
mu_c = 3.0
mu_p = 2.0
g = 9.81
m_pole = 1.0
m_cart = 3.0
MAX_FORCE = 10.0


# State:
class PoleCartState:

  def __init__(self, x, v, theta, theta_dot):
    self.x = x
    self.v = v
    self.theta = theta % (2 * pi)
    self.theta_dot = theta_dot

  def __str__(self):
    return (f"X: {self.x}, V: {self.v}, "
            f"Theta: {self.theta}, Theta Dot: {self.theta_dot}")


# Action: $a \in [-MAX_FORCE, MAX_FORCE]$


# All policies $\pi(s)$ are deterministic.
def move_nothing(state):
  return 0.0


def move_constant(state):
  return 3.0


def move_random(state):
  return 10 * (random.random() - 0.5)


def move_opposite(state):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.theta - pi)
  return -angle * 10


def move_opposite_upswing(state):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.theta - pi)
  if abs(angle) > pi * 0.5:
    return -angle * 10
  return angle * 10


# TODO: IMPLEMENT Q LEARNING - WORK IN PROGRESS


# Use the y component above zero as reward.
def reward(state):
  return max(0, cos(state.theta))


# Simple linear combination of the state params for Q.
def q_function(state, action, params):
  return state.x * params[0] + state.v * params[1] + state.theta * params[
      2] + state.theta_dot * params[3]


# Returns the action (=force) with the best Q value for the given state.
def q_policy(state, q_function, params):
  best_force = 0
  best = q_function(state, 0, params)

  # To simplify, just use a few discrete options for the force.
  for force in [-3, -2, -1, 0, +1, +2, +3]:
    if q_function(state, force, params) > best:
      best = q_function(state, force, params)
      best_force = force

  return best_force


def update_q_function():
  pass


def state_derivative(state, force):
  sin_theta = sin(state.theta)
  cos_theta = cos(state.theta)

  a = m_pole * g * sin_theta * cos_theta
  a -= 7 / 3 * (force + m_pole * l * state.theta_dot**2 * sin_theta -
                mu_c * state.v)
  a -= mu_p * state.theta_dot * cos_theta / l
  a /= m_pole * cos_theta * cos_theta - 7 / 3 * (m_pole + m_cart)

  theta_dd = 3 / (7 * l) * (g * sin_theta - a * cos_theta -
                            mu_p * state.theta_dot / (m_pole * l))

  return (state.theta_dot, theta_dd, state.v, a)


def time_step(state, force, eps=0.0001):
  _, theta_dd, _, a = state_derivative(state, force=force)
  theta = state.theta + state.theta_dot * eps + 1 / 2 * theta_dd * eps**2
  theta_d = state.theta_dot + theta_dd * eps
  x = state.x + state.v * eps + 1 / 2 * a * eps**2
  v = state.v + a * eps

  return PoleCartState(x=x, v=v, theta=theta, theta_dot=theta_d)


def draw(step, state, force):
  x_end = state.x + l * sin(state.theta)
  y_end = -l * cos(state.theta)

  def xx(x_in):
    return 200 + int(x_in * 10)

  def yy(y_in):
    return 50 + int(y_in * 10)

  im = Image.new(mode="RGB", size=(400, 100))
  draw = ImageDraw.Draw(im)

  draw.text((0, 0), f"step={step: >5d}")

  draw.text((0, 70), f"f ={force: >+5.1f}N", fill=(255, 0, 0, 255))
  draw.text((0, 80), f"x ={state.x: >+5.1f}m")
  draw.text((0, 90), f"x'={state.v: >+5.1f}m/s")

  draw.text((80, 80), f"t ={state.theta/pi*180: >+6.1f}°")
  draw.text((80, 90), f"t'={state.theta_dot/pi*180: >+6.1f}°/s")

  draw.line((xx(state.x - 1), yy(0), xx(state.x + 1), yy(0)),
            fill=(255, 255, 255, 128))

  draw.line((xx(state.x), yy(1), xx(state.x + force), yy(1)),
            fill=(255, 0, 0, 128))

  draw.line((xx(x_end), yy(y_end), xx(state.x), yy(0)),
            fill=(255, 255, 255, 128))

  return im


def evaluate(start_state, policy, image_fn=None):
  step = 0
  images = []
  state = start_state
  steps_up = 0
  while True:
    step += 1

    force = policy(state)
    force = max(min(force, MAX_FORCE), -MAX_FORCE)

    state = time_step(state, force=force, eps=0.001)
    if state.theta < 0.1 * pi or state.theta > 1.9 * pi:
      steps_up += 1
    if step % 250 == 0 and image_fn:
      images.append(draw(step, state, force))
    if step > 20000:
      break

  if image_fn:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return steps_up


start_state = PoleCartState(x=0.0, v=0.0, theta=0.01 * pi, theta_dot=0.0)

evaluate(start_state, policy=move_nothing, image_fn="move_nothing.gif")
evaluate(start_state, policy=move_constant, image_fn="move_constant.gif")
evaluate(start_state, policy=move_random, image_fn="move_random.gif")
evaluate(start_state, policy=move_opposite, image_fn="move_opposite.gif")
evaluate(start_state,
         policy=move_opposite,
         image_fn="move_opposite_upswing.gif")

start_state = PoleCartState(x=0.0, v=0.0, theta=0.01 * pi, theta_dot=0.5)

evaluate(start_state, policy=move_nothing, image_fn="move_nothing2.gif")
evaluate(start_state, policy=move_constant, image_fn="move_constant2.gif")
evaluate(start_state, policy=move_random, image_fn="move_random2.gif")
evaluate(start_state, policy=move_opposite, image_fn="move_opposite2.gif")
evaluate(start_state,
         policy=move_opposite,
         image_fn="move_opposite_upswing2.gif")
