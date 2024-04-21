from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")


def prompt_gen(prompts_fn):

  def gen():
    pairs = set()
    for l in open(prompts_fn):
      l = l.strip()
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

      s = s.capitalize()
      t = t.capitalize()
      if (s, t) not in pairs and (t, s) not in pairs:
        pairs.add((s, t))
        yield {"source_text": f"Negate:\n{s}", "target_text": f"{t}"}
        yield {"source_text": f"Negate:\n{t}", "target_text": f"{s}"}

  return gen


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
