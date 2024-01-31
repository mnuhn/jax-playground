from PIL import Image, ImageDraw, ImageFont
import numpy as np
import jax
import io

class Visualizer():
  def __init__(self, width=10000, height=10000):
    self.im = Image.new(mode="RGB", size=(width, height))
    self.id = ImageDraw.Draw(self.im)
    self.box_width=8

  def byte_array(self):
    img_byte_arr = io.BytesIO()
    self.im.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

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
    x_max = 0
    y_max = 0

    font = ImageFont.load_default()
    for k in debug.keys():
      if k == "conv_reshaped" or k == "conv_reshaped_with_nonconv_features":
        continue
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
      x1, y1 = self.box(to_draw, x0, y0)

      self.id.text((x1 + self.box_width, y0), k, font = font, fill=(255, 255, 255, 128))
      text_box = font.getbbox(k)

      x1 = x1 + 2*self.box_width + text_box[2]
      y1 = max(y0 + text_box[3] + self.box_width, y1)

      x_max = max(x_max, x1)
      y_max = max(y_max, y1)

      if k.startswith("conv_") and (k.endswith("conv") or k.endswith("relu")): # TODO or k == "input_conv":
        x0 = x1 + self.box_width
      else:
        x0 = 0
        y0 = y1 + self.box_width

    self.im = self.im.crop((0,0,x_max,y_max))


  def box(self, arr, x0, y0):
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        v = int(arr[i,j])
        self.id.rectangle([(x0+i*self.box_width,y0+j*self.box_width), (x0+(i+1)*self.box_width,y0+(j+1)*self.box_width)], fill=(v,v,v))
    self.id.rectangle([(x0,y0), (x0+arr.shape[0]*self.box_width,y0+arr.shape[1]*self.box_width)], outline='red',width=1)

    return (x0 + arr.shape[0] * self.box_width, y0 + arr.shape[1] * self.box_width)
