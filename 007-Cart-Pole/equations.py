from math import sin, cos, pi
import tqdm
from PIL import Image, ImageDraw

# Static
f = 0
l = 3
mu_c = 3.0
mu_p = 2.0
g = 9.81
m_pole = 1.0
m_cart = 3.0

# Dynamic
theta = 0.01*pi
theta_d = 0
x = 0
x_d = 0

def dot(theta, theta_d, x, x_d):
  sin_theta = sin(theta)
  cos_theta = cos(theta)
  
  x_dd = m_pole*g*sin_theta*cos_theta
  x_dd -= 7/3 * (f + m_pole*l*theta_d**2*sin_theta - mu_c * x_d)
  x_dd -= mu_p * theta_d * cos_theta / l
  x_dd /= m_pole * cos_theta * cos_theta - 7/3 * (m_pole+m_cart)

  theta_dd = 3/(7*l) * (g * sin_theta - x_dd * cos_theta - mu_p * theta_d / (m_pole*l))

  return (theta_d, theta_dd, x_d, x_dd)


def update(theta, theta_d, x, x_d, eps = 0.0001):
  _, theta_dd, _, x_dd = dot(theta, theta_d, x, x_d)
  theta += theta_d * eps + 1/2 * theta_dd * eps**2
  theta_d += theta_dd * eps
  x += x_d * eps + 1/2 * x_dd * eps**2 
  x_d += x_dd * eps

  return theta, theta_d, x, x_d

def draw(frame, theta, x):

  x_end = x + l * sin(theta)
  y_end = - l * cos(theta)

  def xx(x_in):
   return 200 + int(x_in * 10)

  def yy(y_in):
    return 50 + int(y_in*10)

  im = Image.new(mode="RGB", size=(400, 100))
  draw = ImageDraw.Draw(im)

  draw.line((xx(x-1), yy(0), xx(x+1), yy(0)), fill=(255, 255, 255, 128))
  draw.line((xx(x+x_end), yy(y_end), xx(x), yy(0)), fill=(255, 255, 255, 128))

  return im

frame = 0
images = []

while True:
  frame += 1
  print(frame)
  theta, theta_d, x, x_d = update(theta, theta_d, x, x_d)
  theta %= 2*pi
  if frame % 2500 == 0:
    print(x)
    images.append(draw(frame, theta, x))
  if frame > 300000:
    break

images[0].save("frames.gif", save_all=True, append_images=images[1:], duration=10, loop=0)
