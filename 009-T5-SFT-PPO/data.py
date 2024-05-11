from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")


def prompt_gen(prompts_fn):

  def gen():
    for l in open(prompts_fn):
      l = l.strip()
      if len(l) > 80:  # chars
        continue
      yield {"source_text": f"Negate:\n{l}"}

  return gen


def train_gen(training_data):

  def gen():
    pairs = set()
    for l in open(training_data):
      fields = l.strip().split()
      if len(fields) != 2:
        continue
      #assert len(fields) == 2
      s = fields[0].replace("_", " ")
      t = fields[1].replace("_", " ")

      if (s, t) not in pairs and (t, s) not in pairs:
        pairs.add((s, t))
        yield {"source_text": f"Negate:\n{s}", "target_text": f"{t}"}
        yield {"source_text": f"Negate:\n{t}", "target_text": f"{s}"}

      s = s.lower()
      t = t.lower()
      if (s, t) not in pairs and (t, s) not in pairs:
        pairs.add((s, t))
        yield {"source_text": f"Negate:\n{s}", "target_text": f"{t}"}
        yield {"source_text": f"Negate:\n{t}", "target_text": f"{s}"}

  return gen


def tokenize_pairwise_function(example):

  def add_tokens(text, suffix):
    tokenized = tokenizer(text,
                          truncation=True,
                          max_length=128,
                          padding='max_length',
                          return_tensors="pt")

    example[f'input_ids{suffix}'] = tokenized.input_ids
    example[f'attention_mask{suffix}'] = tokenized.attention_mask

  add_tokens(text=example['prompt_text'], suffix='_prompt')
  add_tokens(text=example['accepted_text'], suffix='_chosen')
  add_tokens(text=example['rejected_text'], suffix='_rejected')

  print("prompt_text", example['prompt_text'][0])
  print("mask", example['attention_mask_prompt'][0])
  print("ids", example['input_ids_prompt'][0])

  print("prompt_text", example['accepted_text'][0])
  print("mask", example['attention_mask_chosen'][0])
  print("ids", example['input_ids_chosen'][0])

  return example


def tokenize_input_function(example):
  input_tokenized = tokenizer(example['source_text'],
                              truncation=True,
                              max_length=128,
                              return_tensors="pt")

  example['input_ids'] = input_tokenized.input_ids
  example['attention_mask'] = input_tokenized.attention_mask

  return example


def tokenize_function(example):
  input_tokenized = tokenizer(example['source_text'],
                              padding="max_length",
                              truncation=True,
                              max_length=128,
                              return_tensors="pt")
  target_tokenized = tokenizer(example['target_text'],
                               padding="max_length",
                               truncation=True,
                               max_length=128,
                               return_tensors="pt")

  example['input_ids'] = input_tokenized.input_ids
  example['attention_mask'] = input_tokenized.attention_mask
  example['labels'] = target_tokenized.input_ids

  return example
