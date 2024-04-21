import string
import re
import collections


def jaccard_reward(input_str, output_str, skip_words=set(), case=True):
  input_str = re.sub(r"[,.;@#?!&$]+", ' ', input_str)
  output_str = re.sub(r"[,.;@#?!&$]+", ' ', output_str)

  if not case:
    input_str = input_str.lower()
    output_str = output_str.lower()

  input_set = set(input_str.split()) - skip_words
  output_set = set(output_str.split()) - skip_words
  if len(input_set) == 0 and len(output_set) == 0:
    return 0.0
  jaccard = len(input_set.intersection(output_set)) / (len(
      input_set.union(output_set)))
  return jaccard


def jaccard_non_common_reward(input_str, output_str, case=True):
  skip_words = set([
      'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
      'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
      'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
      'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
      'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
      'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
      'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
      'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
      'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
      'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
      'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
  ])

  return jaccard_reward(input_str, output_str, skip_words)


def punctuation_reward(input_str, output_str):
  input_str = re.sub(r"[A-Za-z ']+", '', input_str).lower()
  output_str = re.sub(r"[A-Za-z ']+", '', output_str).lower()
  if len(input_str) == 0 and len(output_str) == 0:
    return 1.0
  if input_str == output_str:
    return 1.0
  if len(input_str) == 0 or len(output_str) == 0:
    return 0.0
  if input_str[-1] == output_str[-1]:
    return 0.5
  return 0.0


def preprocess(in_str):
  in_str = in_str.strip().lower()
  in_str = re.sub(r"[,.;@#?!&$]+", ' ', in_str)
  in_str = re.sub(r"[ \t]+", ' ', in_str)
  return in_str


def repeated_words_reward(input_str, output_str):

  def most_common_cnt(in_str):
    in_cnts = collections.Counter(preprocess(in_str).split())
    in_most_common = in_cnts.most_common()
    if len(in_most_common) == 0:
      return 0
    return in_most_common[0][1]

  input_repetitions = most_common_cnt(input_str) > 1
  output_repetitions = most_common_cnt(output_str) > 1
  if input_repetitions == output_repetitions:
    return 1.0
  return 0.0


def equality_reward(input_str, output_str):
  input_str = preprocess(input_str)
  output_str = preprocess(output_str)
  if input_str == output_str:
    return 0.0
  return 1.0


def length_reward(input_str, output_str):
  # 1.0 if same length
  # 0.0 if differs
  l_in = len(input_str)
  l_out = len(output_str)

  diff = abs(l_in - l_out) / (l_in + l_out)
  return 1.0 - diff


def ppo_reward(input_str, output_str):
  input_str = input_str.replace("Negate:", "")
  output_str = output_str.replace("Negate:", "")

  result = 0.0

  factor_a = length_reward(input_str, output_str)
  factor_a *= equality_reward(input_str, output_str)
  factor_a *= punctuation_reward(input_str, output_str)

  factor_b = jaccard_reward(input_str, output_str, case=True)
  factor_b += 0.1 * jaccard_reward(input_str, output_str, case=False)
  factor_b += repeated_words_reward(input_str, output_str)

  result = factor_a * factor_b

  return result


def overall_reward(input_str, output_str):
  score = 0.0
  score += 4 * equality_reward(input_str, output_str)
  score += punctuation_reward(input_str, output_str)
  score += jaccard_reward(input_str, output_str)
  score += jaccard_non_common_reward(input_str, output_str)
  score += repeated_words_reward(input_str, output_str)
  return score
