import torch
from torch.nn.functional import log_softmax
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
parser.add_argument('--prompts',
                    dest='prompts',
                    default=None,
                    help='file with prompts')

args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model)
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
  pbar.set_description(f"Processing: '{input_str}'")

  in_tok = data.tokenizer(prompt_str,
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
