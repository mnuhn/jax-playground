antonyms = {}

for l in open("rulebased_reward_model.pairs.txt"):
  s, t = l.strip().lower().split()
  if s == "can" or t == "can":
    continue
  if s == "for" or t == "for":
    continue
  if s not in antonyms:
    antonyms[s] = set()
  if t not in antonyms:
    antonyms[t] = set()
  antonyms[s].add(t)
  antonyms[t].add(s)


def get_not(tokenizer):
  words = [
      "not", "no", "none", "never", "neither", "can't", "couldn't", "wouldn't",
      "won't", "didn't", "don't", "cannot"
  ]
  words.extend([w.capitalize() for w in words])
  words_ids = tokenizer(words, add_special_tokens=False).input_ids
  return words_ids


def remove_subset_lists(list_of_lists):
  # Sort lists by length (longer lists are potential supersets)
  sorted_lists = sorted(list_of_lists, key=len, reverse=True)
  result = []

  # Function to check if the first list is a subset of the second list
  def is_subset(smaller, larger):
    return all(item in larger for item in smaller)

  # Keep each list only if it is not a subset of any existing list in result
  for current_list in sorted_lists:
    subset_found = False
    for result_list in result:
      if is_subset(current_list, result_list):
        subset_found = True
        break
    if not subset_found:
      result.append(current_list)

  # Reverse sort was used, so we reverse the result for consistency with input order
  return result[::-1]


def get_antonyms(input_str, tokenizer):
  words = []
  for x in input_str.replace(".", "").replace(",", "").replace("?", "").replace(
      "!", "").replace(":", "").split():
    if x in antonyms:
      for y in antonyms[x]:
        words.append(y)
        words.append(y.capitalize())
        words.append(y.lower())
        words.append(y.upper())
  words = list(set(words))

  if len(words) > 0:
    print("Force words:", words)
    words_ids = tokenizer(words, add_special_tokens=False).input_ids
    words_ids = remove_subset_lists(words_ids)
    words_ids = [words_ids]
  else:
    print("NO force words")
    words_ids = None

  return words_ids
