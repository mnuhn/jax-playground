import flax.linen as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

import flax
import argparse
import numpy as np
import jax
from jax import lax
import jax.nn
from jax import grad, jit, random
import jax.numpy as jnp
import optax
from tqdm import tqdm
import datalib
import os

p = argparse.ArgumentParser(description='...')
p.add_argument('--timeseries', type=str)
p.add_argument('--history', type=int, default=65)
p.add_argument(
    '--features',
    type=str,
    default="wind_speed,gust_speed,wind_dir,air_pressure,air_temp,water_temp")
p.add_argument('--prediction_files', type=str, default=None)
p.add_argument('--prediction_labels', type=str, default=None)
p.add_argument('--predictions', type=int, default=12)
p = p.parse_args()

assert p.prediction_files
assert p.prediction_labels
assert len(p.prediction_files.split(',')) == len(p.prediction_labels.split(','))

X, Y, XT, YT = datalib.get_data(p.timeseries, ["wind_speed"],
                                history=p.history,
                                predictions=p.predictions,
                                permute=False)

all_predictions = []
for fn in p.prediction_files.split(","):
  all_predictions.append(np.load(fn))

last_predictions = []

COLORS = [
    (1.0, 0.0, 0.0),
    (0.2, 1.0, 0.2),
    (0.2, 0.2, 1.0),
    (0.7, 0.2, 0.7),
    (0.2, 0.7, 0.7),
    (0.7, 0.7, 0.2),
    (0.6, 0.3, 0.3),
    (0.3, 0.6, 0.3),
    (0.3, 0.3, 0.6),
]


def draw(x_values_history, history, x_values_prediction, predictions,
         last_predictions, fn):
  plt.clf()
  plt.rcParams['figure.dpi'] = 250

  # Set up the figure and subplots outside the loop
  fig, axs = plt.subplots(1, 1, figsize=(6, 2), sharex=True)
  c = 0.0

  axs.clear()
  #plt.title(f'Features: {p.features}')
  axs.xaxis.set_major_locator(ticker.MultipleLocator(24))
  axs.set_ylabel("Wind Speed", rotation=90, labelpad=15)
  axs.set_xlabel("Time (h)", rotation=0, labelpad=15)

  y_max = 0.1
  y_max = max(y_max, max(history))
  for i in range(0, len(predictions)):
    y_max = max(y_max, max(predictions[i]))

  for i, (pred_x, pred_y) in enumerate(reversed(last_predictions)):
    c = (i / len(last_predictions))**0.4

    for j in reversed(range(len(pred_y))):
      y_max = max(y_max, max(pred_y[j]))
      if j == 0:
        w = 1.0
        alpha = 0.5 * c
      else:
        w = 0.1
        alpha = 0.2 * c

      axs.plot(pred_x, pred_y[j], linewidth=w, color=COLORS[j], alpha=c)

  axs.plot(x_values_history, history, linewidth=2, color=(0.0, 0.5, 0.0))
  labels = p.prediction_labels.split(',')
  for i in reversed(range(0, len(predictions))):
    label = labels[i]
    w = 1.0
    if i == 0:
      w = 2.0
    axs.plot(x_values_prediction,
             predictions[i],
             linewidth=w,
             color=COLORS[i],
             label=label,
             zorder=3)

  rect = patches.Rectangle((x_values_prediction[0], 0.0),
                           100,
                           1.0,
                           linewidth=1,
                           edgecolor='r',
                           facecolor='white',
                           zorder=2,
                           alpha=0.8)
  axs.add_patch(rect)
  axs.axvline(x_values_prediction[0], color=(0, 0, 0), zorder=3)

  x_ticks = [
      tick for tick in axs.get_xticks() if tick < x_values_prediction[-12]
  ]
  axs.set_xticks(x_ticks)
  axs.set_xlim(x_values_history[0], x_values_prediction[-1])
  axs.set_ylim(bottom=0, top=y_max)
  axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.savefig(fn, bbox_inches='tight')
  plt.close(fig)


for i in range(0, X.shape[0]):
  history = X[i, :]
  predictions = [x[i] for x in all_predictions]

  x_values_history = (np.arange(len(X[i, :, 0])) + i) / 6.0
  x_values_prediction = (np.arange(len(
      predictions[0]))) / 6.0 + x_values_history[-1]
  draw(x_values_history,
       history,
       x_values_prediction,
       predictions,
       last_predictions,
       fn=f'./png/{i:04d}.png')

  last_predictions.insert(0, [x_values_prediction, predictions])
  last_predictions = last_predictions[:100]
