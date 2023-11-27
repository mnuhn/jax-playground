"Prints a confusion matrix for MNIST."
from collections import defaultdict

def print_confusions(iteration, preds, labels, total, loss):
  confusion_matrix = defaultdict(int)
  num_correct = 0
  for cur_label, cur_pred in zip(labels, preds):
    confusion_matrix[(cur_label, cur_pred)] += 1
    if cur_label == cur_pred:
      num_correct += 1

  print("label >   " + " ".join([ f"{c:4}" for c in range(0,10)]))
  print("pred. v +" + "-----" * 10)
  for i in range(0,10):
    confusions = []
    for j in range(0,10):
      confusions.append(confusion_matrix[(i,j)])
    print(f"{i:4}    | " + " ".join([ f"{c:4}" for c in confusions]))

  correct = 0
  for i in range(0,10):
    correct += int(confusion_matrix[(i,i)])

  print(f"iteration={iteration} correct={correct} total={total} loss={loss}")
