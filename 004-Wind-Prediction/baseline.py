import data
import model
import argparse
import numpy as np

p = argparse.ArgumentParser(description='...')
p.add_argument('--data', type=str)
p = p.parse_args()

def report_result(pred, Y, name=None):
  loss = np.mean((Y-pred)**2)**0.5
  print(f"{name} {loss:.08f}")


with np.load(p.data) as data:
  X = data['x_train']
  Y = data['y_train']
  XT = data['x_test']
  YT = data['y_test']

  last_value = np.expand_dims(XT[:,-1,0], axis=(1,))
  mean_value = np.expand_dims(np.mean(XT[:,:,0], axis=1), axis=(1,))
  zero_value = np.zeros(shape=YT.shape)

  report_result(last_value, YT, "last_value")
  report_result(mean_value, YT, "mean_value")
  for v in range(0,10):
    v = v / 100.0
    report_result(zero_value+v, YT, f"const_value_{v:.02f}")

