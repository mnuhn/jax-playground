import sentencepiece as spm

m_unigram = spm.SentencePieceProcessor(model_file='./m_unigram.model')
m_bpe = spm.SentencePieceProcessor(model_file='./m_bpe.model')

inputs = [
    'king',
    'ring',
    'ringo',
    'ringo starr',
    'erring',
    'erroring',
    'terrorizing',
    'doing',
    'wing',
    'ping',
    'inkling',
    'song',
    'bong',
    'nitting',
    'ingo',
    'bingo',
]


def print_model(m, title):
  print()
  print(f"# {title}:")
  for i in inputs:
    print(i, "->", "|".join(m.encode(i, out_type=str)))


print_model(m_unigram, "unigram")
print_model(m_bpe, "bpe")
