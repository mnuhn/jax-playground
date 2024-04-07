from tqdm import tqdm
import random
import jaxopt
from jax import grad
import numpy as np

import state
import q_policy
import draw
import argparse

p = argparse.ArgumentParser(description='...')
p.add_argument('--max_episode_steps', type=int, default=1000)
p.add_argument('--replay_buffer_max_keep', type=int, default=50000)
p.add_argument('--replay_buffer_max_total', type=int, default=75000)
p.add_argument('--num_episodes', type=int, default=250)
p.add_argument('--num_rl_epochs', type=int, default=50)
p.add_argument('--rl_alpha', type=float, default=1.0)
p.add_argument('--rl_gamma', type=float, default=0.0)
p.add_argument('--rl_explore_prob_init', type=float, default=1.0)
p.add_argument('--rl_explore_prob_decay', type=float, default=0.95)
p.add_argument('--q_learning_rate', type=float, default=0.001)
p.add_argument('--q_batch_size', type=int, default=128)
p.add_argument('--q_test_split_frac', type=float, default=0.1)
p.add_argument('--q_model_dim', type=int, default=50)
p.add_argument('--q_patience_epochs', type=int, default=3)

p = p.parse_args()


# Action: $a \in [-MAX_FORCE, MAX_FORCE]$
def evaluate(start_state, policy, params, image_fn=None, explore_prob=0.0):
  step = 0
  images = []
  cur_state = start_state

  states = []
  new_states = []
  action_idxs = []
  q_vals = []

  old_cell = None

  while True:
    step += 1
    states.append(cur_state)
    cell = q_policy.idxs(np.expand_dims(cur_state.vec, axis=0), params)[0]

    # TODO: This is only needed for learning the discrete Q table?!
    #       The simulation code should be independent of the discretization...
    if old_cell is None or not np.array_equal(cell, old_cell):
      action_index, q_val = policy(cur_state,
                                   params=params,
                                   explore_prob=explore_prob)
    #print("step", cur_state, action_index, q_val)
    state_new = cur_state.time_step(force=state.ACTIONS[action_index])

    new_states.append(state_new)
    action_idxs.append(action_index)
    q_vals.append(q_val)

    if step % 25 == 0 and image_fn:
      images.append(
          draw.draw(step, cur_state, force=state.ACTIONS[action_index]))
    if step > p.max_episode_steps or cur_state.reward(action_index,
                                                      state_new) < 0.001:
      break
    cur_state = state_new
    old_cell = cell

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
  for cur_episode in tqdm(range(num_episodes)):
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
    for k in range(len(cur_states)):
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
                                  theta=0.15 * 3.14,
                                  theta_dot=0.0)

r1 = start_state.reward(None, start_state.time_step(force=state.MAX_FORCE))
r2 = start_state.reward(None, start_state.time_step(force=-state.MAX_FORCE))

print()
print(f"Reward delta from {start_state} between good and bad decision:",
      abs(r1 - r2))
print()

explore_prob = p.rl_explore_prob_init

params = q_policy.random_params(p.q_model_dim)
buffered_episodes = []

for epoch in range(p.num_rl_epochs):
  print(f"epoch={epoch} explore_prob={explore_prob}, gamma={p.rl_gamma}")

  print("shuffle data")
  random.shuffle(buffered_episodes)

  if len(buffered_episodes) > p.replay_buffer_max_keep:
    print(
        f"limit past data ({len(buffered_episodes)}) to {p.replay_buffer_max_keep}"
    )
    buffered_episodes = buffered_episodes[:p.replay_buffer_max_keep]

  new_episodes = run_episodes(epoch,
                              num_episodes=p.num_episodes,
                              params=params,
                              explore_prob=explore_prob)

  buffered_episodes = buffered_episodes + new_episodes
  random.shuffle(buffered_episodes)

  if len(buffered_episodes) > p.replay_buffer_max_total:
    print(
        f"limit overall data ({len(buffered_episodes)}) to {p.replay_buffer_max_total}"
    )
    buffered_episodes = buffered_episodes[:p.replay_buffer_max_total]
  print("done")

  gamma = p.rl_gamma
  alpha = p.rl_alpha
  print(f"gamma={gamma} alpha={alpha}")
  print("Compute improved q values")
  improved_q_vec = []
  for ii in tqdm(range(len(buffered_episodes))):
    q_new = q_policy.improved_q_value(cur_state=buffered_episodes[ii][0],
                                      action_idx=buffered_episodes[ii][1],
                                      state_new=buffered_episodes[ii][2],
                                      gamma=gamma,
                                      params=params)
    improved_q_vec.append([q_new])

  s_vecs = np.stack([entry[0].vec for entry in buffered_episodes], axis=0)
  a_idxs = np.stack([entry[1] for entry in buffered_episodes], axis=0)
  sn_vecs = np.stack([entry[2].vec for entry in buffered_episodes], axis=0)
  q_vecs = np.stack(improved_q_vec, axis=0)

  print("Optimizing Q")
  params = q_policy.optimize_model(
      epoch,
      s_vecs,
      a_idxs,
      q_vecs,
      alpha=p.rl_alpha,
      # TODO: Make this compatible with deep q policy.
      #test_split_frac=p.q_test_split_frac,
      #learning_rate=p.q_learning_rate,
      #batch_size=p.q_batch_size,
      #patience_epochs=p.q_patience_epochs,
      params=params)

  explore_prob *= p.rl_explore_prob_decay
