import random

def randomly_merge_successive_words(sentence):
    words = sentence.split()
    if len(words) < 2:
        return sentence

    # Select a random index to merge with the next word
    index = random.randint(0, len(words) - 2)

    # Merge the selected word with the next word
    merged_word = words[index] + words[index + 1]

    # Replace the selected word with the merged word and remove the next word
    words[index] = merged_word
    del words[index + 1]

    return ' '.join(words)
