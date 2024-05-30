# https://medium.com/@martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638

import os
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments

import torch.nn as nn
import torch
import random
import data
import sys
import time
import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import Adafactor
from transformers import default_data_collator
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
                    default=None,
                    help='which model to open')
parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    default=0.001,
                    type=float,
                    help='which model to open')
parser.add_argument('--margin',
                    dest='margin',
                    default=0.0,
                    type=float,
                    help='which model to open')
parser.add_argument('--weight_decay',
                    dest='weight_decay',
                    default=0.05,
                    type=float,
                    help='which model to open')
parser.add_argument('--epochs',
                    dest='epochs',
                    default=20.0,
                    type=float,
                    help='which model to open')
parser.add_argument('--out_model',
                    dest='out_model',
                    default=None,
                    help='which model to open')
parser.add_argument('--tb_suffix',
                    dest='tb_suffix',
                    default=str(random.randint(10000, 99999)),
                    help='which model to open')
parser.add_argument('--suffix',
                    dest='suffix',
                    default="",
                    help='which model to open')
parser.add_argument('--test_frac',
                    dest='test_frac',
                    default=0.1,
                    type=float,
                    help='how much data to use for the test set')
parser.add_argument('--train_all_parameters',
                    dest='train_all_parameters',
                    default=None,
                    help='which model to open')

args = parser.parse_args()

db = prompt_db.prompt_db(args.db)

all_data = Dataset.from_generator(
    db.get_preference_pairs_gen()).shuffle(seed=42)

all_data = all_data.map(data.tokenize_pairwise_function,
                        load_from_cache_file=False,
                        batched=True)  #, batched=True)

dataset = all_data.train_test_split(test_size=args.test_frac,
                                    shuffle=True,
                                    seed=45)

for x in dataset["test"]:
  print(x["prompt_text"])
  print(x["accepted_text"])
  print(x["rejected_text"])

if not args.model:
  sys.exit(1)
if not args.out_model:
  args.out_model = args.model + "-reward"

model = T5ForSequenceClassification.from_pretrained(args.model)

for name, param in model.named_parameters():
  print(name, param.size())
  if 'classification' in name:
    param.requires_grad = True
  else:
    if args.train_all_parameters:
      param.requires_grad = True
    else:
      param.requires_grad = False

os.makedirs(f'{args.out_model}{args.suffix}.tensorboard/', exist_ok=True)
log_dir = f'{args.out_model}{args.suffix}.tensorboard/{args.tb_suffix}'
writer = SummaryWriter(log_dir=log_dir)
print(f"logging tensorboard to {log_dir}")


class CustomRewardTrainer(RewardTrainer):

  def compute_loss(
      self,
      model: Union[PreTrainedModel, nn.Module],
      inputs: Dict[str, Union[torch.Tensor, Any]],
      return_outputs=False,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

    rewards_chosen = model(
        input_ids=inputs["input_ids_prompt"],
        attention_mask=inputs["attention_mask_prompt"],
        decoder_input_ids=inputs["input_ids_chosen"],
        decoder_attention_mask=inputs["attention_mask_chosen"],
        return_dict=True,
    )["logits"][:1]
    rewards_rejected = model(
        input_ids=inputs["input_ids_prompt"],
        attention_mask=inputs["attention_mask_prompt"],
        decoder_input_ids=inputs["input_ids_rejected"],
        decoder_attention_mask=inputs["attention_mask_rejected"],
        return_dict=True,
    )["logits"][:1]

    loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

    # calculate loss, optionally modulate with margin
    if "margin" in inputs:
      loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected -
                                       inputs["margin"]).mean()
    else:
      loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected -
                                       args.margin).mean()

    if return_outputs:
      return loss, {
          "rewards_chosen": rewards_chosen,
          "rewards_rejected": rewards_rejected,
      }
    return loss


class CustomLoggingCallback(TrainerCallback):

  def __init__(self, log_every_x_steps=100):
    self.log_every_x_steps = log_every_x_steps

  def on_log(self, args, state, control, logs=None, **kwargs):
    if logs is not None:
      for k, v in logs.items():
        writer.add_scalar(k, v, state.global_step)

  def on_step_end(self, args, state, control, model=None, **kwargs):
    if state.global_step % self.log_every_x_steps != 0:
      return
    if model is None:
      return

    count = 0
    for e in dataset["test"]:
      count += 1
      if count > 10:
        break

      def reward(text):
        tok = data.tokenizer(text,
                             padding='max_length',
                             truncation=True,
                             max_length=128,
                             return_tensors="pt")
        outputs = model(input_ids=tok.input_ids.to(model.device))

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][0].item(), probs[0][1].item()

      #print(outputs.logits)

      print("------------------")
      print(e["accepted_text"], reward(e["accepted_text"]))
      print(e["rejected_text"], reward(e["rejected_text"]))


training_args = RewardConfig(
    output_dir=args.out_model,
    num_train_epochs=args.epochs,
    warmup_steps=50,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=args.weight_decay,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_strategy='steps',
    evaluation_strategy='steps',
    save_steps=20,
    eval_steps=20,
    remove_unused_columns=False,
    max_length=128,
)

trainer = CustomRewardTrainer(
    model=model,
    args=training_args,
    tokenizer=data.tokenizer,
    data_collator=default_data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    callbacks=[CustomLoggingCallback(log_every_x_steps=100)],
)

trainer.train()
trainer.save_model(args.out_model + "-final")
