import q_policy
import state
import numpy as np
from math import pi
import argparse
import random

p = argparse.ArgumentParser(description='...')
p.add_argument('--q_learning_rate', type=float, default=0.001)
p.add_argument('--q_batch_size', type=int, default=128)
p.add_argument('--q_test_split_frac', type=float, default=0.1)
p.add_argument('--q_model_dim', type=int, default=100)
p.add_argument('--q_patience_epochs', type=int, default=3)
p.add_argument('--num_examples', type=int, default=50000)

p = p.parse_args()

epoch = 1
cached_episodes = []
improved_q_vec = []

for i in range(p.num_examples):
  a = 0  #i % 2
  cur_state = state.PoleCartState(x=0, v=0, theta=0.0, theta_dot=0.0)
  next_state = cur_state.time_step(state.ACTIONS[a])
  r = cur_state.reward(a, next_state) + a
  if i < 3:
    print(next_state, r)
  cached_episodes.append((cur_state, a, cur_state, r))
  improved_q_vec.append([r])

s_vecs = np.stack([entry[0].vec for entry in cached_episodes], axis=0)
a_idxs = np.stack([entry[1] for entry in cached_episodes], axis=0)
sn_vecs = np.stack([entry[2].vec for entry in cached_episodes], axis=0)
q_vecs = np.stack(improved_q_vec, axis=0)

params = q_policy.optimize_model(
    epoch,
    s_vecs,
    a_idxs,
    q_vecs,
    #test_split_frac=p.q_test_split_frac,
    #learning_rate=p.q_learning_rate,
    #batch_size=p.q_batch_size,
    #patience_epochs=p.q_patience_epochs,
    params=q_policy.random_params(p.q_model_dim))
