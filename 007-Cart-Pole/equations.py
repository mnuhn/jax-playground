from tqdm import tqdm
import random
import jaxopt
from jax import grad
import numpy as np

import state
import q_policy
import draw

# Static
MAX_STEP = 1000

MAX_STEP_KEEP = 75000
MAX_STEP_MORE = 25000

NUM_EPISODES = 50
NUM_EPOCHS = 50


# Action: $a \in [-MAX_FORCE, MAX_FORCE]$
def evaluate(start_state, policy, params, image_fn=None, explore_prob=0.0):
  step = 0
  images = []
  cur_state = start_state

  states = []
  new_states = []
  action_idxs = []
  improved_q_vec = []

  while True:
    step += 1
    states.append(cur_state)

    action_index = policy(cur_state, params=params, explore_prob=explore_prob)
    state_new = cur_state.time_step(force=state.ACTIONS[action_index])

    new_states.append(state_new)
    action_idxs.append(action_index)

    if step % 25 == 0 and image_fn:
      images.append(
          draw.draw(step, cur_state, force=state.ACTIONS[action_index]))
    if step > MAX_STEP:  # abs(cur_state.vec[INDEX_THETA] - pi) < 0.5 * pi or
      break
    cur_state = state_new

  if image_fn and len(images) > 0:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return step, states, action_idxs, new_states


def run_episodes(iteration, num_episodes, params, explore_prob):
  reward_avg = 0
  states = []
  a_idxs = []
  new_states = []
  q_vecs = []
  print(f"Iteration {iteration}")
  for cur_episode in tqdm(range(0, num_episodes)):
    start_state = state.random_state()
    if cur_episode < 5:
      image_fn = f"gifs/q_policy{iteration}.{cur_episode}.gif"
    else:
      image_fn = None
    episode_steps, cur_states, cur_action_idxs, cur_new_states = evaluate(
        start_state,
        policy=q_policy.q_policy_noval,
        params=params,
        image_fn=image_fn,
        explore_prob=explore_prob)

    states.extend(cur_states)
    a_idxs.extend(cur_action_idxs)
    new_states.extend(cur_new_states)

    r_episode = 0.0
    for k in range(0, len(cur_states)):
      r = states[k].reward(a_idxs[k], new_states[k])
      r_episode += r
    r_episode /= len(cur_states)

    reward_avg += r_episode

  print(f"{iteration}: Average reward {reward_avg / num_episodes}")

  return states, a_idxs, new_states


start_state = state.PoleCartState(x=0.0,
                                  v=0.0,
                                  theta=0.25 * 3.14,
                                  theta_dot=0.0)
next_state = start_state.time_step(force=10)
print("reward1:", start_state.reward(None, next_state))

next_state = start_state.time_step(force=-10)
print("reward2:", start_state.reward(None, next_state))

all_states = []
all_a_idxs = []
all_new_states = []

explore_prob = 1.0
gamma = 0.0

params = q_policy.random_params()

for epoch in range(0, NUM_EPOCHS):
  print(f"epoch={epoch} explore_prob={explore_prob}, gamma={gamma}")
  next_states, next_a_idxs, next_new_states = run_episodes(
      epoch,
      num_episodes=NUM_EPISODES,
      params=params,
      explore_prob=explore_prob)

  print("shuffle data")
  c = list(zip(all_states, all_a_idxs, all_new_states))
  random.shuffle(c)

  if len(c) > MAX_STEP_KEEP:
    print(f"limit past data ({len(c)}) to {MAX_STEP_KEEP}")
    c = c[:MAX_STEP_KEEP]

  d = list(zip(next_states, next_a_idxs, next_new_states))
  random.shuffle(d)

  if len(d) > MAX_STEP_MORE:
    print(f"limit overall data ({len(d)}) to {MAX_STEP_MORE}")
    d = d[:MAX_STEP_MORE]

  e = c + d

  all_states, all_a_idxs, all_new_states = [list(x) for x in zip(*e)]
  print("done")

  s_vecs = np.stack([cur_state.vec for cur_state in all_states], axis=0)
  a_idxs = np.stack(all_a_idxs, axis=0)
  sn_vecs = np.stack([cur_state.vec for cur_state in all_new_states], axis=0)

  improved_q_vec = []
  for ii in range(len(all_states)):
    q_new = q_policy.improved_q_value(all_states[ii],
                                      all_a_idxs[ii],
                                      all_new_states[ii],
                                      gamma=gamma,
                                      params=params)
    improved_q_vec.append(q_new)
  improved_q_vec = np.stack(improved_q_vec, axis=0)

  print("Optimizing Q")
  params = q_policy.optimize_model(epoch,
                                   s_vecs,
                                   a_idxs,
                                   improved_q_vec,
                                   params=params)
  explore_prob *= 0.95
  if epoch >= 10 and epoch % 10 == 0:
    gamma += 0.05
