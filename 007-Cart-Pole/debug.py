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
p.add_argument('--q_model_dim', type=int, default=10)
p.add_argument('--q_patience_epochs', type=int, default=3)

p = p.parse_args()

epoch = 1
cached_episodes = []
improved_q_vec = []

for i in range(1000):
  cur_state = state.random_state()
  a = i % 2
  cached_episodes.append((cur_state, a, cur_state, 1.0))
  improved_q_vec.append([cur_state.vec[state.INDEX_COS_THETA]])

s_vecs = np.stack([entry[0].vec for entry in cached_episodes], axis=0)
a_idxs = np.stack([entry[1] for entry in cached_episodes], axis=0)
sn_vecs = np.stack([entry[2].vec for entry in cached_episodes], axis=0)
q_vecs = np.stack(improved_q_vec, axis=0)

params = q_policy.optimize_model(epoch,
                                 s_vecs,
                                 a_idxs,
                                 q_vecs,
                                 test_split_frac=p.q_test_split_frac,
                                 learning_rate=p.q_learning_rate,
                                 batch_size=p.q_batch_size,
                                 patience_epochs=p.q_patience_epochs,
                                 params=q_policy.random_params(p.q_model_dim))
