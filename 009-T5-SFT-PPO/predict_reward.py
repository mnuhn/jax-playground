import torch
from torch.nn.functional import log_softmax
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
import prompt_db
from tqdm import tqdm
import skip_file

import argparse
import rulebased_reward_model as reward_model

parser = argparse.ArgumentParser(description='add predictions to database')
parser.add_argument('--model',
                    dest='model',
                    default="training/1713207876-final",
                    help='which model to open')
parser.add_argument('--db',
                    dest='db',
                    default=None,
                    help='db to write predictions to')
parser.add_argument('--skip_file',
                    dest='skip_file',
                    default=None,
                    help='prompts to skip')
parser.add_argument('--num_per_prompt',
                    dest='num_per_prompt',
                    default=1,
                    type=int,
                    help='number of completions per prompt')
parser.add_argument('--prompts',
                    dest='prompts',
                    default=None,
                    help='file with prompts')

args = parser.parse_args()

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

candidates = []
skip_set = skip_file.skipSet(args.skip_file)

db = prompt_db.prompt_db(args.db)

pbar = tqdm(open(args.prompts, "r"))
for l in pbar:
  input_str = l.strip()
  if skip_set.should_skip(input_str):
    continue
  prompt_id = db.add_prompt(input_str)
  prompt_str = f"Negate:\n{input_str}"
  input_ids = tokenizer.encode(prompt_str,
                               padding=True,
                               truncation=True,
                               max_length=128,
                               return_tensors="pt")
  input_ids = input_ids.to(model.device)

  pbar.set_description(f"Processing: '{input_str}'")

  outputs = model.generate(
      input_ids,
      return_dict_in_generate=True,
      output_scores=True,
      max_length=20,  #temperature=0.9, do_sample=True
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
    reward = reward_model.overall_reward(input_str, output_str)
    db.add_completion(prompt_id, output_str, reward, overall_score)
  db.conn.commit()
