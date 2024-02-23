# Bradley-Terry demo with NBA stats.

import numpy as np
import jax.numpy as jnp
import jax.nn
import csv
from jax import random
import optax

def read_games(fn, reserve=500):
  team_ids = {}
  team_names = []
  counts = np.zeros(shape=(reserve,reserve))

  def add_team(t):
    if t not in team_ids:
      team_ids[t] = len(team_ids)
      team_names.append(t)
    return team_ids[t]
    
  with open(fn) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)

    for t1, t2, _, wl in reader:
      t1 = add_team(t1)
      t2 = add_team(t2)
      win = 0
      if wl == "W":
        win = 1
      counts[t1,t2] += win 

  assert(len(team_ids) < reserve)
  counts = counts[:len(team_ids), :len(team_ids)]
  np.set_printoptions(threshold=np.inf)
  return team_ids, team_names, counts

key = random.PRNGKey(0)

# We have data N_{i<j} and N_{j<i} which counts how often in a match "i vs j" i
# (or j) won.
team_ids, team_names, counts = read_games('nba-2012.csv')

# Do matches between pairs (i,j) with i,j \in 1..I..

# The goal is to assign a weight $\beta_i$ to each item, such that the
# likelihood of i winning against j can be estimated as:

def loss(counts, betas):
  # p(i>j) = exp(\beta_i) / ( exp \beta_i + exp \beta_j )
  #        = 1 / ( 1 + exp [ \beta_j - \beta_i ] 
  #        = sigmoid(\beta_i - \beta_j)
  # LL = \sum_i \sum_j N_{i>j} \log p(i<j)
  # LL = \sum_i \sum_j N_{i>j} \log \sigmoid (\beta_i - \beta_j)
  delta = jnp.transpose(jnp.atleast_2d(betas)) - betas
  delta = jax.nn.sigmoid(delta)
  delta = jnp.fill_diagonal(delta, 1, inplace=False)
  delta = - jnp.log(delta)

  res = jnp.sum(jnp.multiply(counts, delta))

  # Regularization: choose "small" betas if possible.
  res += 0.01 * jnp.sum(jnp.abs(betas)) 

  return res

betas = random.normal(key,shape=(len(team_ids),))

optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(betas)

betas_grad = jax.grad(loss, argnums=1)

for step in range(0,10000):
  cur_betas_grad = betas_grad(counts, betas)
  updates, opt_state = optimizer.update(cur_betas_grad, opt_state, betas)
  betas = optax.apply_updates(betas, updates)

  if step % 100 == 0:
    print(step, loss(counts, betas))
    print("========")
    team_ranks = np.argsort(-betas)
    for rank, team_id in enumerate(team_ranks):
      print(f'{rank+1}. {team_names[int(team_id)]} ({betas[team_id]:.2f})')
