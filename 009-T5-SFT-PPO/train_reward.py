# https://medium.com/@martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
import torch
import random
import data
import sys
import time
import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForSequenceClassification, T5ForConditionalGeneration, T5Config
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

from transformers import TrainingArguments

from trl import RewardTrainer, RewardConfig

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset

import prompt_db

parser = argparse.ArgumentParser(description='Train SFT')
parser.add_argument('--db',
                    dest='db',
                    default=None,
                    help='db to write predictions to')
parser.add_argument('--model',
                    dest='model',
                    default="training/1713207876-final",
                    help='which model to open')

args = parser.parse_args()

db = prompt_db.prompt_db(args.db)

all_data = Dataset.from_generator(
    db.get_preference_pairs_gen()).shuffle(seed=42)
dataset = all_data.train_test_split(test_size=0.1, shuffle=True, seed=45)
dataset = dataset.map(data.tokenize_pairwise_function,
                      load_from_cache_file=False,
                      batched=True)  #, batched=True)

model = T5ForSequenceClassification.from_pretrained(args.model)

out_fn = f'./training/{str(int(time.time()))}'
print(f"writing data to {out_fn}")


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
      in_tok = data.tokenizer(e["source_text"],
                              padding='max_length',
                              truncation=True,
                              max_length=128,
                              return_tensors="pt")

      def reward(text):
        dec_tok = data.tokenizer(text,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=128,
                                 return_tensors="pt")
        outputs = model(input_ids=in_tok.input_ids.to(model.device),
                        decoder_input_ids=dec_tok.input_ids.to(model.device),
                        decoder_attention_mask=dec_tok.attention_mask.to(
                            model.device))

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][0].item(), probs[0][1].item()

      #print(outputs.logits)

      print("------------------")
      print(e["source_text"])
      print(e["accepted_text"], reward(e["accepted_text"]))
      print(e["rejected_text"], reward(e["rejected_text"]))


training_args = RewardConfig(
    output_dir=out_fn,
    num_train_epochs=25.0,
    warmup_steps=10,
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_strategy='steps',
    evaluation_strategy='steps',
    save_steps=100,
    eval_steps=10,
    remove_unused_columns=False,
    max_length=128,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=data.tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    callbacks=[CustomLoggingCallback(log_every_x_steps=50)],
)

trainer.train()
trainer.save_model(out_fn + "-reward")
