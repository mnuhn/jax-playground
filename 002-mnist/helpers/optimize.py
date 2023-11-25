"Swappable optimizers to be able to compare my own implementation to optax's."
import optax
import jax.numpy as jnp

class SelfSgd:
  """Self-implemented SGD."""
  def __init__(self, model, learning_rate, beta, decay, decay_step):
    self.model = model
    self.g = [jnp.zeros_like(model.params[i]) for i in range(
        0, len(model.params))]
    self.learning_rate = learning_rate
    self.beta = beta
    self.decay = decay
    self.decay_step = decay_step
    self.step = 0

  def update(self, cur_images, cur_labels):
    self.step += 1
    if self.step % int(self.decay_step) == 0:
      self.learning_rate = self.decay * self.learning_rate
      print("new learning rate:", self.learning_rate)
    cur_g = self.model.gradient(self.model.params,cur_images,cur_labels)
    self.g = [ self.beta * self.g[i] + (
        1.0-self.beta) * cur_g[i] for i in range(0, len(self.model.params))]
    self.model.params = [ self.model.params[i] - (
        self.learning_rate * self.g[i]) for i in range(
            0, len(self.model.params)) ]

class OptaxSgd:
  """Wrapper for Optax SGD."""
  def __init__(self, model, learning_rate):
    self.model = model
    self.tx = optax.sgd(learning_rate=learning_rate)
    self.opt_state = self.tx.init(model.params)

  def update(self, cur_images, cur_labels):
    g_params = self.model.gradient(self.model.params,cur_images,cur_labels)
    updates, self.opt_state = self.tx.update(g_params, self.opt_state)
    self.model.params = optax.apply_updates(self.model.params, updates)

class OptaxAdam:
  """Wrapper for Optax ADAM."""
  def __init__(self, model, learning_rate):
    self.model = model
    self.tx = optax.adam(learning_rate=learning_rate)
    self.opt_state = self.tx.init(model.params)

  def update(self, cur_images, cur_labels):
    g_params = self.model.gradient(self.model.params,cur_images,cur_labels)
    updates, self.opt_state = self.tx.update(g_params, self.opt_state)
    self.model.params = optax.apply_updates(self.model.params, updates)

