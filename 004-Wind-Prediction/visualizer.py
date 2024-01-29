from PIL import Image, ImageDraw
import numpy as np

class Visualizer():
  def __init__(self,width,height):
    self.im = Image.new(mode="RGB", size=(width, height))
    self.id = ImageDraw.Draw(self.im)

  def save(self, fn):
    self.im.save(fn)

  def box(self, arr, x0, y0, width):
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        v = int(arr[i,j])
        self.id.rectangle([(x0+i*width,y0+j*width), (x0+(i+1)*width,y0+(j+1)*width)], fill=(v,v,v))
    self.id.rectangle([(x0,y0), (x0+arr.shape[0]*width,y0+arr.shape[1]*width)], outline='red',width=1)
