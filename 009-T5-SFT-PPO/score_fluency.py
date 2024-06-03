import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def score_sentence(sentence):

  inputs = tokenizer.encode(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(inputs, labels=inputs)
  print(outputs)
  loss = outputs.loss.item()

  return loss


for s in [
    "I am not here", "I'mn't here", "I am here.", "I don't am here.",
    "I willn't go.", "This is absolutely correct."
]:
  score = score_sentence(s)
  print(f"{s}: {score}")
