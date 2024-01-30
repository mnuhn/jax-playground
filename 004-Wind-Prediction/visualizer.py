from PIL import Image, ImageDraw, ImageFont
import numpy as np
import jax

class Visualizer():
  def __init__(self,width,height):
    self.im = Image.new(mode="RGB", size=(width, height))
    self.id = ImageDraw.Draw(self.im)
    self.box_width=8

  def save(self, fn):
    self.im.save(fn)

  def draw_dict(self, debug, num):
    def scale(p, minimum=None, maximum=None):
      if minimum == None:
        minimum = np.min(p)
      if maximum == None:
        maximum = np.max(p)
      return (p - minimum) / (maximum-minimum+0.001) * 255

    y0 = 0
    x0 = 0
    font = ImageFont.load_default()
    for k in debug.keys():
      to_draw = debug[k][num]
      if len(to_draw.shape) == 1:
        to_draw = np.expand_dims(to_draw, axis=1)
      if to_draw.shape[0] == 0:
        print("error with", to_draw)
        continue
      if to_draw.shape[1] == 0:
        print("error with", to_draw)
        continue

      to_draw = scale(to_draw)
      w, h = self.box(to_draw, x0, y0)
      self.id.text((x0 + w + self.box_width, y0), k, font = font, fill=(255, 255, 255, 128))
      text_box = font.getbbox(k)

      if k.startswith("conv_") and (k.endswith("conv") or k.endswith("relu")):
        x0 += w + 2*self.box_width + text_box[2]
      else:
        x0 = 0
        y0 += h + self.box_width


  def box(self, arr, x0, y0):
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        v = int(arr[i,j])
        self.id.rectangle([(x0+i*self.box_width,y0+j*self.box_width), (x0+(i+1)*self.box_width,y0+(j+1)*self.box_width)], fill=(v,v,v))
    self.id.rectangle([(x0,y0), (x0+arr.shape[0]*self.box_width,y0+arr.shape[1]*self.box_width)], outline='red',width=1)

    return (arr.shape[0] * self.box_width, arr.shape[1] * self.box_width)
