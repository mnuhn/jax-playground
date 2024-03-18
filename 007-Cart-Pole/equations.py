from math import sin, cos, pi
import tqdm
from PIL import Image, ImageDraw

# Static
l = 3
mu_c = 3.0
mu_p = 2.0
g = 9.81
m_pole = 1.0
m_cart = 3.0


def controller(state):
  return 0.0  #random.random()


class PoleCartState:

  def __init__(self, x, v, theta, theta_dot):
    self.x = x
    self.v = v
    self.theta = theta % (2 * pi)
    self.theta_dot = theta_dot

  def __str__(self):
    return (f"X: {self.x}, V: {self.v}, "
            f"Theta: {self.theta}, Theta Dot: {self.theta_dot}")


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


def draw(frame, state):
  x_end = state.x + l * sin(state.theta)
  y_end = -l * cos(state.theta)

  def xx(x_in):
    return 200 + int(x_in * 10)

  def yy(y_in):
    return 50 + int(y_in * 10)

  im = Image.new(mode="RGB", size=(400, 100))
  draw = ImageDraw.Draw(im)

  draw.line((xx(state.x - 1), yy(0), xx(state.x + 1), yy(0)),
            fill=(255, 255, 255, 128))
  draw.line((xx(x_end), yy(y_end), xx(state.x), yy(0)),
            fill=(255, 255, 255, 128))

  return im


frame = 0
images = []

state = PoleCartState(x=0.0, v=0.0, theta=0.01 * pi, theta_dot=0.0)

while True:
  frame += 1
  state = time_step(state, force=controller(state))
  if frame % 2500 == 0:
    images.append(draw(frame, state))
  if frame > 300000:
    break

images[0].save("frames.gif",
               save_all=True,
               append_images=images[1:],
               duration=10,
               loop=0)
