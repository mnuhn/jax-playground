# All policies $\pi(s)$ are deterministic.
def move_nothing(state, params=None):
  return 0.0


def move_constant(state, params=None):
  return 3.0


def move_random(state, params=None):
  return 10 * (random.random() - 0.5)


def move_opposite(state, params=None):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.vec[INDEX_THETA] - pi)
  return -angle * 10


def move_opposite_upswing(state, params=None):
  # If the pole falls to the right, move right - and vice versa.
  angle = (state.vec[INDEX_THETA] - pi)
  if abs(angle) > pi * 0.5:
    return -angle * 10
  return angle * 10
