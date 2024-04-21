class skipSet:

  def __init__(self, filename):
    """Build a set of prompts to skip based on file in format:
            Prompt_str | Response_str.
    """
    self.skip_set = set()
    for l in open(filename, "r"):
      l = l.strip()

      s = l.split()[0]
      s = s.replace("_", " ")
      self.skip_set.add(s)
      self.skip_set.add(s.lower())

      t = l.split()[1]
      t = s.replace("_", " ")
      self.skip_set.add(t)
      self.skip_set.add(t.lower())

  def should_skip(self, line):
    if line.strip().lower() in self.skip_set:
      return True
    return False
