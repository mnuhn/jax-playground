import string
import re
import collections


def combine(rule_reward, reward, rule_reward_fac, override):
  total_reward = (1.0 -
                  rule_reward_fac) * reward + rule_reward_fac * rule_reward
  if override and rule_reward < 1.75:
    total_reward *= 0.8
  if override and rule_reward < 1.5:
    total_reward *= 0.8
  if override and rule_reward < 1.0:
    total_reward *= 0.8
  if override and rule_reward < 0.5:
    total_reward *= 0.8
  if override and rule_reward < 0.1:
    total_reward *= 0.1
    total_reward -= 0.5
  return total_reward


antonyms = {}

for l in open("rulebased_reward_model.pairs.txt"):
  s, t = l.strip().lower().split()
  if s not in antonyms:
    antonyms[s] = set()
  if t not in antonyms:
    antonyms[t] = set()
  antonyms[s].add(t)
  antonyms[t].add(s)


def jaccard_reward(input_str, output_str, skip_words=set(), case=True):
  input_str = re.sub(r"[,.;@#?!&$]+", ' ', input_str)
  output_str = re.sub(r"[,.;@#?!&$]+", ' ', output_str)

  if not case:
    input_str = input_str.lower()
    output_str = output_str.lower()

  input_set = set(input_str.split()) - skip_words
  output_set = set(output_str.split()) - skip_words

  input_set_extended = set(input_set)
  for y in input_set:
    if y in antonyms:
      for x in antonyms[y]:
        input_set_extended.add(x)

  if len(input_set) == 0 and len(output_set) == 0:
    return 0.1
  jaccard = len(input_set_extended.intersection(output_set)) / (len(
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
    return 0.1
  if input_str[-1] == output_str[-1]:
    return 0.5
  return 0.1


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
  return 0.1


def equality_reward(input_str, output_str):
  input_str = preprocess(input_str)
  output_str = preprocess(output_str)
  if input_str == output_str:
    return 0.01
  return 1.0


def length_reward(input_str, output_str):
  # 1.0 if same length
  # 0.0 if differs
  l_in = len(input_str)
  l_out = len(output_str)

  diff = abs(l_in - l_out) / (l_in + l_out)
  result = 1.0 - diff
  # allow some wiggle room
  if result > 0.8:
    return 1.0
  return 0.1


IS_HAS_DOES_DID = "(is|was|were|has|had|am|are|do|does|did|can|shall|must|might|should|won't)"
IS_HAS = "(am|is|was|were|has|had|am|were|are)"
DOES_DID = "(do|does|did|can|shall|won't)"
NOT_REGEX = re.compile("n't")
WORKS_WITH_NOT_REGEX = re.compile(
    "(is|was|were|has|had|are|do|does|did|can|shall|wo|must|might|should)")
ADJECTIVE = "(good|bad|nice|terrible|friendly|big|small|wide|narrow|here|there|open|close|[a-z]*ing)"
REGEXES = [
    re.compile(f'\\b{DOES_DID}(n\'t)?[a-z]{2,}'),
    re.compile(f'\\b{DOES_DID}(n\'t)? {IS_HAS}\\b'),
    re.compile(f'\\b{DOES_DID}(n\'t)? ([a-z]*s)\\b'),
    re.compile(f'\\b(am|will|\'m|go)n\'t\\b'),
    re.compile(f'n\'t[a-z]+\\b'),
    re.compile(f'\\b{IS_HAS} {DOES_DID}(n\'t)? '),
    re.compile(f'\\b(not) {IS_HAS}\\b'),
    re.compile(f'\\b(an|a) (the)\\b'),
    re.compile(f'\\b(a) ([aeiou][a-z]*)\\b'),
    re.compile(f'\\ba {ADJECTIVE}([.?!])?$'),
    re.compile(r'  ')
]


def broken_grammar(input_str, output_str):
  input_set = set(input_str.lower().split())
  output_str = output_str.lower()
  if NOT_REGEX.search(
      output_str) and not WORKS_WITH_NOT_REGEX.search(output_str):
    return 0.1
  for r in REGEXES:
    if r.search(output_str):
      return 0.1

  for i in input_set:
    if "not" + i in output_str:
      return 0.1
    if i + "not" in output_str:
      return 0.1
    if i + "n't" in output_str:
      return 0.1
    if "n't" + i in output_str:
      return 0.1
  return 1.0


NOT = re.compile(f'(\\bnot|n\'t)( |$)')


def avoid_not(output_str):
  output_str = output_str.lower()
  if NOT.search(output_str):
    return 0.8
  return 1.0


def ppo_reward(input_str, output_str):
  input_str = input_str.replace("Negate:", "").strip()
  output_str = output_str.replace("Negate:", "").strip()

  result = 0.0

  factor_a = length_reward(input_str, output_str)
  factor_a *= equality_reward(input_str, output_str)
  factor_a *= punctuation_reward(input_str, output_str)
  factor_a *= broken_grammar(input_str, output_str)
  factor_a *= avoid_not(output_str)

  factor_b = jaccard_reward(input_str, output_str, case=True)**0.3
  factor_b += 0.1 * jaccard_reward(input_str, output_str, case=False)**0.3
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
