import torch
from torch.nn.functional import log_softmax
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
import prompt_db
from tqdm import tqdm
import random

import argparse

parser = argparse.ArgumentParser(description='add predictions to database')
parser.add_argument('--model',
                    dest='model',
                    default="training/1713207876-final",
                    help='which model to open')
parser.add_argument('--db',
                    dest='db',
                    default=None,
                    help='db to write predictions to')
parser.add_argument('--num_per_prompt',
                    dest='num_per_prompt',
                    default=1,
                    type=int,
                    help='number of completions per prompt')
parser.add_argument('--max_len',
                    dest='max_len',
                    default=50,
                    type=int,
                    help='maximum length')
parser.add_argument('--prompts',
                    dest='prompts',
                    default=None,
                    help='file with prompts')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

candidates = []

input_str = None
force_words_str = None

for cur_input in sys.stdin:
  cur_input = cur_input.strip()

  if input_str is None:
    input_str = cur_input
    continue

  if force_words_str is None:
    force_words_str = cur_input

  prompt_str = f"Negate:\n{input_str}"
  input_ids = tokenizer.encode(prompt_str,
                               padding='max_length',
                               truncation=True,
                               max_length=128,
                               return_tensors="pt")
  input_ids = input_ids.to(model.device)

  force_words = force_words_str.split()
  force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids
  print("Generating for:")
  print(input_str)
  print(force_words_str)
  print("===")

  input_str = None
  force_words_str = None

  outputs = model.generate(
      input_ids,
      return_dict_in_generate=True,
      output_scores=True,
      max_length=args.max_len,
      force_words_ids=force_words_ids,
      temperature=1.0,
      remove_invalid_values=True,
      #do_sample=True,
      num_beams=args.num_per_prompt,
      num_return_sequences=args.num_per_prompt,
      no_repeat_ngram_size=2,
      repetition_penalty=1.5)

  transition_scores = model.compute_transition_scores(outputs.sequences,
                                                      outputs.scores,
                                                      outputs.beam_indices,
                                                      normalize_logits=False)
  output_length = np.sum(transition_scores.numpy() < 0, axis=1)
  length_penalty = model.generation_config.length_penalty
  reconstructed_scores = transition_scores.sum(axis=1) / (output_length**
                                                          length_penalty)
  for i in range(len(outputs.sequences)):
    output_str = tokenizer.decode(outputs.sequences[i],
                                  skip_special_tokens=True)
    input_length = input_ids.shape[1]
    generated_tokens = outputs.sequences[i, 0:]
    overall_score = reconstructed_scores[i].item()

    print(f"{overall_score:.4f} {output_str}")
