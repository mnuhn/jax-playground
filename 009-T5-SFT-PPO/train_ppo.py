# https://medium.com/@martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import random
import data
import numpy as np
from tqdm import tqdm
import time
import argparse

from trl import PPOTrainer
from trl import PPOConfig
from trl.models import AutoModelForSeq2SeqLMWithValueHead

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset

#import rulebased_reward_model as reward_model

DEVICE = "cuda:0"

parser = argparse.ArgumentParser(description='Train PPO')
parser.add_argument('--model',
                    dest='model',
                    default="training/1713207876-final",
                    help='which model to open')
parser.add_argument('--reward_model',
                    dest='reward_model',
                    default="training/1713732543-reward",
                    help='which model to open')
parser.add_argument('--prompts_fn',
                    dest='prompts_fn',
                    default=None,
                    help='prompts_fn')
parser.add_argument('--max_ppo_steps',
                    dest='max_ppo_steps',
                    default=20,
                    type=int,
                    help='max_ppo_steps')

args = parser.parse_args()

dataset = Dataset.from_generator(data.prompt_gen(
    args.prompts_fn)).shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
dataset = dataset.map(data.tokenize_input_function,
                      load_from_cache_file=False)  #, batched=True)
dataset = dataset.remove_columns(["source_text"])


def collator(data):
  result = {key: [d[key] for d in data] for key in data[0]}
  return result


ppo_config = PPOConfig(
    model_name=args.model + ".ppo",
    learning_rate=0.00002,
    target_kl=1.0,
    batch_size=128,
    #use_score_scaling=True,
    mini_batch_size=8,
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    args.model).to(DEVICE)
model.eval()

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    dataset=dataset["train"],
    data_collator=collator,
    #data_collator=data_collator,
    tokenizer=data.tokenizer,
)

generation_kwargs = {
    #    "min_length": 5,
    #    "max_length": 50,
    "max_new_tokens": 64,
    "top_k": 0.0,
    "top_p": 1.0,
    #"pad_token_id": tokenizer.eos_token_id,
    #    "pad_token_id": -1, #data.tokenizer.pad_token_id,
    #    "eos_token_id": -1, # data.tokenizer.eos_token_id,
    #"eos_token_id": -1,
    "do_sample": True,
    "begin_suppress_tokens": [data.tokenizer.eos_token_id],
    #"return_prompt": False,
    #"no_repeat_ngram_size": 3,
    "batch_size": 1,
}


def pad_sequences(sequences, max_length):
  padding_needed = [max_length - seq.size(0) for seq in sequences]
  padded_sequences = torch.stack([
      torch.nn.functional.pad(seq, (0, pad), value=data.tokenizer.pad_token_id)
      if pad > 0 else seq for seq, pad in zip(sequences, padding_needed)
  ])

  return padded_sequences


for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
  if step >= args.max_ppo_steps:
    break

  prompt_tensors = [torch.tensor(p[0]) for p in batch["input_ids"]]
  prompt_strs = [
      data.tokenizer.decode(p[0], skip_special_tokens=True)
      for p in batch["input_ids"]
  ]
  batch["query"] = prompt_strs

  response_tensors = ppo_trainer.generate(prompt_tensors,
                                          return_prompt=False,
                                          **generation_kwargs)

  batch["response"] = data.tokenizer.batch_decode(response_tensors,
                                                  skip_special_tokens=True)

  padded_response_tensors = pad_sequences(response_tensors, max_length=256)
  padded_input_ids = pad_sequences(prompt_tensors, max_length=256)

  reward_tensors = []
  q_num = 0
  for q, r in zip(padded_input_ids, padded_response_tensors):
    outputs_attention_mask = (r != data.tokenizer.pad_token_id).int()
    reward_outputs = reward_model(
        input_ids=q.unsqueeze(0).to(reward_model.device),
        decoder_input_ids=r.unsqueeze(0).to(reward_model.device),
        decoder_attention_mask=outputs_attention_mask.unsqueeze(0).to(
            reward_model.device))
    reward_probs = torch.nn.functional.softmax(reward_outputs.logits, dim=-1)
    reward_tensors.append(torch.tensor(reward_probs[0][1]))
    q_num += 1

  # Run PPO step.
  stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
  ppo_trainer.log_stats(stats, batch, reward_tensors)

  print(f'objective/kl: {stats["objective/kl"]}')
  print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
  print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
  print('-'.join('' for x in range(100)))

out_fn = f'./training/{str(int(time.time()))}.ppo'
ppo_trainer.save_pretrained(out_fn)
