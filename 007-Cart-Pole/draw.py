import state
from PIL import Image, ImageDraw
from math import sin, cos, pi


def draw(step, cur_state, force):
  x_end = cur_state.vec[state.INDEX_X] + state.const_l * sin(
      cur_state.vec[state.INDEX_THETA])
  y_end = -state.const_l * cos(cur_state.vec[state.INDEX_THETA])

  def xx(x_in):
    return 200 + int(x_in * 10)

  def yy(y_in):
    return 50 + int(y_in * 10)

  im = Image.new(mode="RGB", size=(400, 100))
  draw = ImageDraw.Draw(im)

  draw.text((0, 0), f"step={step: >5d}")

  draw.text((0, 70), f"f ={force: >+5.1f}N", fill=(255, 0, 0, 255))
  draw.text((0, 80), f"x ={cur_state.vec[state.INDEX_X]: >+5.1f}m")
  draw.text((0, 90), f"x'={cur_state.vec[state.INDEX_V]: >+5.1f}m/s")

  draw.text((80, 80), f"t ={cur_state.vec[state.INDEX_THETA]/pi*180: >+6.1f}°")
  draw.text((80, 90),
            f"t'={cur_state.vec[state.INDEX_THETA_DOT]/pi*180: >+6.1f}°/s")

  draw.line((xx(cur_state.vec[state.INDEX_X] - 1), yy(0),
             xx(cur_state.vec[state.INDEX_X] + 1), yy(0)),
            fill=(255, 255, 255, 128))

  draw.line((xx(cur_state.vec[state.INDEX_X]), yy(1),
             xx(cur_state.vec[state.INDEX_X] + force / 10.0), yy(1)),
            fill=(255, 0, 0, 128))

  draw.line((xx(x_end), yy(y_end), xx(cur_state.vec[state.INDEX_X]), yy(0)),
            fill=(255, 255, 255, 128))

  return im
