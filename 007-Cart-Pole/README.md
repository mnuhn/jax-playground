# Cart Pole Physics

Based on https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html

For Experiments, scroll down.

## Trajectories for simple policy
I show two examples:

1. Pole is slightly out of balance, but without any angular speed
2. Pole is slightly out of balance, with some angular speed

### Noop policy
This policy does nothing: the force on the cart is always 0.

<img src="move_nothing.gif">
<img src="move_nothing2.gif">

### Constant force policy
This policy applies a constant force to the cart.

<img src="move_constant.gif">
<img src="move_constant2.gif">

### Random force policy
This policy applies a random force to the cart.

<img src="move_random.gif">
<img src="move_random2.gif">

### Move opposite policy
This policy applies a force based on the pole's angle. If it's "on the
right", apply a force to the right, and vice versa.

<img src="move_opposite.gif">
<img src="move_opposite2.gif">

### Move opposite (improved) policy
Similar to the previous one, but try to "swing the pole up" when it's in the
lower half.

<img src="move_opposite_upswing.gif">
<img src="move_opposite_upswing2.gif">

## Recap: Markov Decision Process

Based on https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture26-ri.pdf

### Definitions

Markov Decision Process = RL Framework + Markov Assumption

* States $S$, sequence $s_0^T = (s_0, s_1, ..., s_T)$
* Actions $A$, sequence $a_0^T = (a_0, a_1, ..., a_T)$
* Markov assumption: $p(s_{t+1}|s_0^t, a_0^t) = p(s_{t+1}|s_t,a_t)$
* Reward assumption: $r(s_0^{t+1}, a_0^t) = r(s_t, a_t, s_{t+1}) = r_{t+1}$
* Policy: $\pi(s_t|a_t) = p(a_t|s_t)$

Goal: Find the optimal policy
$$
\tilde \pi = \arg\max\limits_{\pi} \sum_{t=0}^{\infty} \gamma^t r_t
$$

Discount rate $\gamma$ is introduced to make the reward finite.

The value function computes the expected discouted rewards of state $s$ when
following policy $\pi$. It can be rewritten by taking into account the Markov
assumption:

$$
\begin{eqnarray}
V^\pi(s) &=& \sum\limits_{a_0^T,s_0^T|s_0=s} p(s_0^T, a_0^T) \sum\limits_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1})\\
         &=& \sum\limits_{a} \pi(a|s) \sum\limits_{s'} p(s'|s,a) [ r(s,a,s') + \gamma V^{\pi}(s')]
\end{eqnarray}
$$

A similar calculation can be done for the Action-Value Function $Q^\pi(s,a)$,
which computes the expected discounted rewards following policy $\pi$ after
taking action $a$ from state $s$:

$$
\begin{eqnarray}
Q^\pi(s,a) &=& \sum\limits_{\stackrel{s_0^T}{s_0=s}}  \sum\limits_{\stackrel{a_0^T}{a_0=a}} p(s_0^T, a_0^T) \sum\limits_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1})\\
           &=& \sum\limits_{a} \pi(a|s) \sum\limits_{s'} p(s'|s,a) [ r(s,a,s') + \gamma \sum\limits_{a'} \pi(a'|s') Q^{\pi}(s',a')]
\end{eqnarray}
$$

### Relationships between $Q$ and $V$
The following relationships hold:
$$
\begin{eqnarray}
V^{\pi}(s) &=& \sum\limits_{a} \pi(a|s) Q^{\pi}(s,a)\\
Q^{\pi}(s,a) &=& \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma V^{\pi}(s') \right]
\end{eqnarray}
$$

Optimal action-value function:
$$
Q^{*}(s,a) = \max\limits_{\pi} Q^{\pi}(s,a)
$$

Bellman optimality equation for $Q^*$:

$$
Q^{*}(s,a) = \sum\limits_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma \max\limits_{a'} Q^{*}(s',a')\right]
$$

TODO: If we have $Q^{*}$, then we can get the optimal policy by using a greedy algorithm ?

### Typical RL Tasks

#### Task: Given a policy $\pi$ what is $V^{\pi}(s)$ and $Q^{\pi}(s,a)$
This can be done by iteratively applying
$$
V^{k+1}(s) = \sum\limits_{a} \pi(a|s) \sum\limits_{s'} p(s'|s,a) [ r(s,a,s') + \gamma V^{k}(s')].
$$

We can start with $V^k(s)$ being arbitrary.

#### Task: Find an improved policy $\pi' > \pi$ such that $V^{\pi'}(s) > V^{\pi}(s) \forall s$
Choose new $\tilde\pi(s) = \arg\max\limits_{a} \sum\limits_{s'} p(s'|s,a) [ r(s,a,s') + \gamma V^{\pi}(s') ]$

(Is that correct??)

#### Task: Find the optimal policy

When we know the models $p(s,a,s')$ and $r(s,a,s')$:

3 Steps:
1. Inititalize some $\pi(s)$
2. Evaluate Policy, i.e. compute $V^{\pi}(s)$
3. Optimize policy: for all states, choose new $\tilde\pi(s) = \arg\max\limits_{a} \sum\limits_{s'} p(s'|s,a) [ r(s,a,s') + \gamma V^{\pi}(s') ]$

## Experiments

UNDER CONSTRUCTION

### Replay Buffer

I implement a simple cache of past episodes. Each episode generates a list of
states, eventually ending after MAX_ITERATIONS or when the reward is 0.

This list of states (and actions) is added to a replay buffer. Before a new
episode is run, the old replay buffer is shuffled and trimmed to
REPLAY_CACHE_MAX_KEEP. Then, the new states are added (from the current
episode). Then the new buffer is shuffled again and trimmed to
REPLAY_CACHE_MAX_TOTAL.

This is to ensure the number of training examples cannot explore, while still
keeping past experiences/episodes available.

### Reward

#### First Try: $r=\cos(\theta)$
I first started of with using $cos(\theta)$ as the immediate reward. I first
started off with Deep-Q Learning (see below) - for which this reward did not
work well (I set $\gamma=0$): In order to pick the right action, the prediction
of the $cos(\theta)$ need to be extremely accurate. I was able to get the model
to predict this reward accurately enough, but it needed a lot of training
interations and some other tricks (predicting only the delta to the current
states angle theta).

#### Second Try: $r=+1$ if upright, $r=0$ otherwise
I then switched to this simpler reward. This is conceptually simpler and worked
well for the discrete q learning (still need to make it work for deep q
learning).

### Discrete Q Learning (discretize the state space)

The idea is to discretize the state space, and to then learn a table function.
While the state space is discretized, the physics operate still on the
non-discrete state space.

When implementing an $\epsilon$-greedy policy, I first applied exploration on
the continuous state space. This meant that in a given "cell", during one
episode, I would do random action for every time step. A key change that made
things work was to decide for a random action once the simulation enters a
"cell", and to then keep that action constant until the simulation enters a new
"cell".

```
python3 equations.py \
  --num_episodes=10 \
  --max_episode_steps=1000 \
  --q_model_dim=90 \
  --rl_gamma=0.8 \
  --rl_alpha=1.0 \
  --rl_explore_prob_init=1.0 \
  --rl_explore_prob_decay=0.6 
```


### Deep-Q Learning

Here, the state-space is not discretized and a neural network is learned that
directly takes the state variables as input and predicts the Q values.

I got something working - but the current state of the code is broken. Some
problems I ran into along the way:

* I first tried to train a model $Q(state, action)$, i.e. the action is an
  input to the network. However, this didn't work well.
* Only then I moved to training models $Q_action(state)$ - so essentially 1 NN
  per possible action.
