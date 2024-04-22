# https://medium.com/@martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import random
import data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
import time
import argparse

parser = argparse.ArgumentParser(description='Train SFT')
parser.add_argument('--training_data',
                    dest='training_data',
                    default=None,
                    help='training_data')

args = parser.parse_args()

model = AutoModelForSeq2SeqLM.from_pretrained(
    "t5-small")  # "training/1712925943-final")#
data_collator = DataCollatorWithPadding(tokenizer=data.tokenizer)

all_data = Dataset.from_generator(data.train_gen(
    args.training_data)).shuffle(seed=42)
dataset = all_data.train_test_split(test_size=0.05, shuffle=True, seed=42)

tokenized_datasets = dataset.map(data.tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ["source_text", "target_text"])


def predict(model, dataset):
  predictions = []
  for batch in dataset:
    pred = model.generate(**batch[0])  # batch[0] contains the input features
    predictions.extend(
        [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred])
  return predictions


class CustomLoggingCallback(TrainerCallback):

  def __init__(self, log_every_x_steps=100):
    self.log_every_x_steps = log_every_x_steps

  def on_step_end(self, args, state, control, model=None, **kwargs):
    if state.global_step % self.log_every_x_steps != 0:
      return
    if model is None:
      return

    count = 0
    for e in dataset["test"]:
      count += 1
      if count >= 5:
        break
      input_ids = data.tokenizer.encode(e["source_text"],
                                        padding=True,
                                        truncation=True,
                                        max_length=128,
                                        return_tensors="pt")
      input_ids = input_ids.to(model.device)
      outputs = model.generate(
          input_ids, max_length=128,
          num_beams=5)  #, early_stopping=True, temperature=0.7)
      debug_in = e["source_text"].replace("Negate:\n", "")
      debug_out = data.tokenizer.decode(outputs[0], skip_special_tokens=True)
      print(f'Example: "{debug_in}" -> "{debug_out}"')


out_fn = f'./training/{str(int(time.time()))}'

optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,
    #lr = 5e-4,
)

training_args = TrainingArguments(
    output_dir=out_fn,
    num_train_epochs=10.0,
    warmup_steps=10,
    learning_rate=None,  #1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_strategy='steps',
    evaluation_strategy='steps',
    save_steps=100,
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    callbacks=[CustomLoggingCallback(log_every_x_steps=100)],
    optimizers=(optimizer, AdafactorSchedule(optimizer)
               )  # (optimizer, lr_scheduler)
)

trainer.train()
trainer.save_model(out_fn + "-final")
