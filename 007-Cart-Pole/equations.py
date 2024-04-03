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

EPISODES_CACHE_MAX = 75000
EPISODES_CACHE_OLD = 50000
MAX_STEP_KEEP = 75000

NUM_EPISODES = 250
NUM_EPOCHS = 50

ALPHA = 1.0


# Action: $a \in [-MAX_FORCE, MAX_FORCE]$
def evaluate(start_state, policy, params, image_fn=None, explore_prob=0.0):
  step = 0
  images = []
  cur_state = start_state

  states = []
  new_states = []
  action_idxs = []
  q_vals = []

  while True:
    step += 1
    states.append(cur_state)

    action_index, q_val = policy(cur_state,
                                 params=params,
                                 explore_prob=explore_prob)
    state_new = cur_state.time_step(force=state.ACTIONS[action_index])

    new_states.append(state_new)
    action_idxs.append(action_index)
    q_vals.append(q_val)

    if step % 25 == 0 and image_fn:
      images.append(
          draw.draw(step, cur_state, force=state.ACTIONS[action_index]))
    if step > MAX_STEP or cur_state.vec[state.INDEX_COS_THETA] < 0.5:
      break
    cur_state = state_new

  if image_fn and len(images) > 0:
    images[0].save(image_fn,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,
                   loop=0)

  return step, states, action_idxs, new_states, q_vals


def run_episodes(iteration, num_episodes, params, explore_prob):
  reward_avg = 0
  steps_avg = 0.0

  states = []
  a_idxs = []
  new_states = []
  q_vals = []
  print(f"Iteration {iteration}")
  for cur_episode in tqdm(range(0, num_episodes)):
    start_state = state.random_upright_state()
    if cur_episode < 5:
      image_fn = f"gifs/q_policy{iteration}.{cur_episode}.gif"
    else:
      image_fn = None
    episode_steps, cur_states, cur_action_idxs, cur_new_states, cur_q_vals = evaluate(
        start_state,
        policy=q_policy.q_policy,
        params=params,
        image_fn=image_fn,
        explore_prob=explore_prob)

    states.extend(cur_states)
    a_idxs.extend(cur_action_idxs)
    new_states.extend(cur_new_states)
    q_vals.extend(cur_q_vals)

    r_episode = 0.0
    for k in range(0, len(cur_states)):
      r = states[k].reward(a_idxs[k], new_states[k])
      r_episode += r
    r_episode /= len(cur_states)

    reward_avg += r_episode
    steps_avg += episode_steps

  print(
      f"{iteration}: Average steps: {steps_avg/num_episodes} - average reward: {reward_avg / num_episodes}"
  )

  return list(zip(states, a_idxs, new_states, q_vals))


start_state = state.PoleCartState(x=0.0,
                                  v=0.0,
                                  theta=0.25 * 3.14,
                                  theta_dot=0.0)
next_state = start_state.time_step(force=10)
print("reward1:", start_state.reward(None, next_state))

next_state = start_state.time_step(force=-10)
print("reward2:", start_state.reward(None, next_state))

gamma = 0.0
explore_prob = 1.0

params = q_policy.random_params()
cached_episodes = []

for epoch in range(0, NUM_EPOCHS):
  print(f"epoch={epoch} explore_prob={explore_prob}, gamma={gamma}")

  print("shuffle data")
  random.shuffle(cached_episodes)

  if len(cached_episodes) > EPISODES_CACHE_OLD:
    print(f"limit past data ({len(cached_episodes)}) to {EPISODES_CACHE_OLD}")
    cached_episodes = cached_episodes[:EPISODES_CACHE_OLD]

  new_episodes = run_episodes(epoch,
                              num_episodes=NUM_EPISODES,
                              params=params,
                              explore_prob=explore_prob)

  cached_episodes = cached_episodes + new_episodes
  random.shuffle(cached_episodes)

  if len(cached_episodes) > EPISODES_CACHE_MAX:
    print(
        f"limit overall data ({len(cached_episodes)}) to {EPISODES_CACHE_MAX}")
    cached_episodes = cached_episodes[:EPISODES_CACHE_MAX]
  print("done")

  improved_q_vec = []
  for ii in range(len(cached_episodes)):
    q_new = q_policy.improved_q_value(cached_episodes[ii][0],
                                      cached_episodes[ii][1],
                                      cached_episodes[ii][2],
                                      gamma=gamma,
                                      params=params)
    q_use = (1 - ALPHA) * cached_episodes[ii][3] + ALPHA * q_new
    improved_q_vec.append([q_use])

  s_vecs = np.stack([entry[0].vec for entry in cached_episodes], axis=0)
  a_idxs = np.stack([entry[1] for entry in cached_episodes], axis=0)
  sn_vecs = np.stack([entry[2].vec for entry in cached_episodes], axis=0)
  q_vecs = np.stack(improved_q_vec, axis=0)

  print("Optimizing Q")
  params = q_policy.optimize_model(epoch,
                                   s_vecs,
                                   a_idxs,
                                   q_vecs,
                                   params=q_policy.random_params())
  explore_prob *= 0.95
  #if epoch >= 10 and epoch % 10 == 0:
  #  gamma *= 0.05
