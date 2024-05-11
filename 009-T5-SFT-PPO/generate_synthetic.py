replacements = []
import re

for l in open("./data/mix16.txt"):
  l = l.strip()
  s = l.split()[0]
  s = s.replace("_", " ")
  t = l.split()[1]
  t = t.replace("_", " ")

  if len(s.split()) > 1:
    continue
  if len(t.split()) > 1:
    continue
  if s == t:
    continue
  if len(s) <= 3 or len(t) <= 3:
    continue
  replacements.append((s, t))
  replacements.append((t, s))

replacements = sorted(list(set(replacements)))

for l in open("./data/medium_sentences.txt"):
  l = l.strip()
  l_new = l
  cur_replacements = []
  for s, t in replacements:
    if s not in l_new:
      continue
    if (t, s) not in cur_replacements:
      l_new = re.sub(r"\b%s\b" % s, t, l_new)
      cur_replacements.append((s, t))
  if l_new != l and len(cur_replacements) == 1:
    print(l.replace(" ", "_"), l_new.replace(" ", "_"))
