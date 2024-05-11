import difflib


def color_diff(str1, str2):
  d = difflib.Differ()
  diff = list(d.compare(str1, str2))

  output = []
  for word in diff:
    if word.startswith('-'):
      output.append(f"\033[41m \033[0m")  # Red background for deletions
    elif word.startswith('+'):
      output.append(
          f"\033[42m{word[2:]}\033[0m")  # Green background for additions
    elif word.startswith(' '):
      output.append(
          f"\033[47m{word[2:]}\033[0m")  # White background for common parts
  return ''.join(output)


# Example usage
str1 = "He went to school."
str2 = "He did not go to school!"

print(color_diff(str1, str2))
