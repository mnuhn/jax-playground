"Train loop"
from tqdm import tqdm
import numpy as np
from helpers import eval_print

def train_loop(model, batches, test_images, test_labels, update, iterations):
  pbar = tqdm(range(iterations))
  cur_loss = None
  for i in pbar:
    cur_images, cur_labels = next(batches)
    update(cur_images, cur_labels)

    if i % int(1+iterations/100) == 0:
      cur_loss = model.loss(model.params,cur_images,cur_labels)
      cur_test_loss = model.test_loss(model.params)
      pbar.set_description(
              f"train_loss={cur_loss:.3f} test_loss={cur_test_loss:.3f}")
      cur_preds = np.argmax(
              model.predict(model.params,test_images), axis=-1).tolist()
      cur_labels = np.argmax(test_labels, axis=-1).tolist()
      eval_print.print_confusions(
              i, cur_preds, cur_labels, test_images.shape[0], cur_loss)
