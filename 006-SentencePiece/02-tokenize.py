import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='./m.model')

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

for i in inputs:
  print(i, "->", "|".join(sp.encode(i, out_type=str)))
