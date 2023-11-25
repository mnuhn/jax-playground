'Parse logs file and print them such that the results can be sorted.'
import glob

# Example file format:
# batch_size-32-learning_rate-0.05-bias_term-true-optimizer-self_sgd-decay-1.0-decay_epochs-100.txt

def parse_line(l):
  l = l.strip()
  lookup = dict()
  if not l.startswith('iteration='):
    return None, None, None
  fields = l.split()
  for field in fields:
    k, v = field.split('=')
    lookup[k] = v
  return lookup['iteration'], lookup['correct'], lookup['loss']

def parse_fn(fn):
  fields = fn.split('-')
  batch_size = int(fields[1])
  learning_rate = float(fields[3])
  bias_term = fields[5]
  optimizer = fields[7]
  decay = 'n/a'
  if optimizer == 'self_sgd':
    decay = fields[9] + '-' + fields[11]
  return batch_size, learning_rate, bias_term, optimizer, decay


for x in glob.glob('*.txt'):
  batch_size, learning_rate, bias_term, optimizer, decay = parse_fn(x)
  f = open(x, 'r')
  iterations = []
  for l in f:
    iteration, correct, loss = parse_line(l)
    if iteration:
      iteration = int(iteration)
      iterations.append([iteration, correct, loss])

  best_iteration, best_correct, best_loss = max(iterations, key=lambda x: x[1])
  best_epoch = best_iteration * batch_size / 60000.0
  print(' | '.join([str(x) for x in [batch_size, learning_rate, bias_term,
                                     optimizer, decay, best_epoch,
                                     best_correct, best_loss]]))
