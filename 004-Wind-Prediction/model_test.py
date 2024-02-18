import jax
import jax.numpy as jnp
import pytest
import model


def test_my_custom_layer():
  print("running test")
  rng = jax.random.PRNGKey(0)

  BATCH_SIZE = 16
  HISTORY = 8
  INPUT_FEATURES = 10
  NUM_CONVS = 1
  NON_CONV_FEATURES = 2
  CONV_CHANNELS = 11
  FEATURES_PER_PREDICTION = 3
  PREDICTIONS = 12

  m = model.CNN(
      channels=CONV_CHANNELS,
      conv_len=5,
      num_convs=NUM_CONVS,
      dense_size=25,
      num_dense=10,
      down_scale=2,
      predictions=PREDICTIONS,
      features_per_prediction=FEATURES_PER_PREDICTION,
      batch_norm=False,
      dropout=0.0,
      padding='SAME',
      nonconv_features=NON_CONV_FEATURES,
  )

  root_key = jax.random.key(seed=0)
  main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
  x_batch = jnp.zeros(shape=[BATCH_SIZE, HISTORY, INPUT_FEATURES])
  x_batch = x_batch.at[0, 1, 1].set(1.0)
  variables = m.init(params_key, x_batch, train=False)
  params = variables['params']

  preds, debug = m.apply({'params': params}, x_batch, train=False, debug=True)

  assert debug["input"].shape == (BATCH_SIZE, HISTORY, INPUT_FEATURES)
  assert debug["input_conv"].shape == (BATCH_SIZE, HISTORY,
                                       INPUT_FEATURES - NON_CONV_FEATURES)

  if NUM_CONVS > 0:
    assert debug["conv_0_conv"].shape == (BATCH_SIZE, HISTORY, CONV_CHANNELS)
  assert debug["final"].shape == (BATCH_SIZE, PREDICTIONS,
                                  FEATURES_PER_PREDICTION)

  for name in debug.keys():
    print(name, debug[name].shape)
