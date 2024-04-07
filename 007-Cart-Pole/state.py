from math import sin, cos, pi
import numpy as np
import random

MAX_FORCE = 25.0
ACTIONS = np.array([-MAX_FORCE, MAX_FORCE])  #arange(-MAX_FORCE, MAX_FORCE)

INDEX_X = 0
INDEX_V = 1
INDEX_THETA = 2
INDEX_THETA_DOT = 3
INDEX_SIN_THETA = 4
INDEX_COS_THETA = 5

const_l = 3
mu_c = 3.0
mu_p = 2.0
const_g = 9.81
m_pole = 1.0
m_cart = 3.0
eps = 0.01


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

  def derivative(self, force):
    a = m_pole * const_g * self.vec[INDEX_SIN_THETA] * self.vec[INDEX_COS_THETA]
    a -= 7 / 3 * (force + m_pole * const_l * self.vec[INDEX_THETA_DOT]**2 *
                  self.vec[INDEX_SIN_THETA] - mu_c * self.vec[INDEX_V])
    a -= mu_p * self.vec[INDEX_THETA_DOT] * self.vec[INDEX_COS_THETA] / const_l
    a /= m_pole * self.vec[INDEX_SIN_THETA] * self.vec[
        INDEX_COS_THETA] - 7 / 3 * (m_pole + m_cart)
    theta_dd = 3 / (7 * const_l) * (const_g * self.vec[INDEX_SIN_THETA] -
                                    a * self.vec[INDEX_COS_THETA] -
                                    mu_p * self.vec[INDEX_THETA_DOT] /
                                    (m_pole * const_l))

    return np.array([self.vec[INDEX_THETA_DOT], theta_dd, self.vec[INDEX_V], a])

  def time_step(self, force):
    _, theta_dd, _, a = self.derivative(force=force)
    theta = self.vec[INDEX_THETA] + self.vec[
        INDEX_THETA_DOT] * eps + 1 / 2 * theta_dd * eps**2
    theta_d = self.vec[INDEX_THETA_DOT] + theta_dd * eps
    x = self.vec[INDEX_X] + self.vec[INDEX_V] * eps + 1 / 2 * a * eps**2
    v = self.vec[INDEX_V] + a * eps

    return PoleCartState(x=x,
                         v=v,
                         theta=theta,
                         theta_dot=theta_d,
                         step=self.step + 1)

  # Use the delta in y component above zero as reward.
  # Use the distance to 0
  def reward(self, action_index, state_new):
    result = pi - abs(state_new.vec[INDEX_THETA] % (2 * pi) - pi)
    if result < pi / 4:
      return 1.0
    else:
      return 0.0
    #result -= 0.1 * abs(state_new.vec[INDEX_X])

    #assert result > 0
    #assert result < pi

    #result = state_new.vec[INDEX_COS_THETA]
    #weight = 0.1 * min(0.0, state_new.vec[INDEX_COS_THETA] - 0.8)
    #result -= weight * state_new.vec[INDEX_THETA_DOT]

  def __str__(self):
    return (
        f"theta: {self.vec[INDEX_THETA]:.3f} theta': {self.vec[INDEX_THETA_DOT]:.3f}"
    )


def random_state():
  return PoleCartState(
      x=0.0,
      v=0.0,  #random.gauss() * 0.1,
      theta=random.random() * 2 * pi,
      theta_dot=random.gauss() * 0.001)


def random_upright_state():
  return PoleCartState(
      x=0.0,
      v=0.0,  #random.gauss() * 0.1,
      theta=random.random() * 0.1 * pi - 0.05 * pi,
      theta_dot=random.gauss() * 0.001)
