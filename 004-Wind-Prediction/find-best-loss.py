"""Reads all tensorboard data files and output the best eval_loss for each"""

from tensorboard.backend.event_processing import event_accumulator
import argparse
import glob
import tqdm
import tensorflow as tf
from tensorflow.python.framework import tensor_util

p = argparse.ArgumentParser(description='...')
p.add_argument('--glob', type=str, required=True)
p = p.parse_args()

def get_best_loss(fn):
  try:
    ea = event_accumulator.EventAccumulator(fn, size_guidance=event_accumulator.DEFAULT_SIZE_GUIDANCE)
    ea.Reload()

    losses = []
    for event in ea.Tensors('eval_loss'):
      loss = tensor_util.MakeNdarray(event.tensor_proto)
      losses.append((event.step, loss))

    return min(losses, key=lambda x: x[1])

  except Exception as e:
    print(e)
    return 1.0, -1

for fn in tqdm.tqdm(glob.glob(p.glob)):
  step, loss = get_best_loss(fn)
  print(fn, step, loss)
