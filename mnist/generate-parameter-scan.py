# Generate some runs to try out.
for batch_size in [32,64,128]:
  for learning_rate in [0.001, 0.005, 0.01, 0.05]:
    for bias_term in ["true", "false"]:
      for optimizer in ["adam", "self_sgd" ]:
        decay = "na"
        decay_epochs = "na"
        if optimizer == "self_sgd":
          epochs=300
          for decay in [0.99, 0.98, 0.95, 0.9]:
            for decay_epochs in [5,10]:
              log_file = f"logs/batch_size-{batch_size}-learning_rate-{learning_rate}-bias_term-{bias_term}-optimizer-{optimizer}-decay-{decay}-decay_epochs-{decay_epochs}.txt"
              print(f"python3 train.py --epochs={epochs} --batch_size={batch_size} --bias_term={bias_term} --learning_rate={learning_rate} --optimizer={optimizer} --learning_rate_decay={decay} --learning_rate_decay_epoch_step={decay_epochs} > {log_file}")
        else:
          epochs=75
          log_file=f"logs/batch_size-{batch_size}-learning_rate-{learning_rate}-bias_term-{bias_term}-optimizer-{optimizer}-decay-{decay}-decay_epochs-{decay_epochs}.txt"
          print(f"python3 train.py --epochs={epochs} --batch_size={batch_size} --bias_term={bias_term} --learning_rate={learning_rate} --optimizer={optimizer} > {log_file}")
