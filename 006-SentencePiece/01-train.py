import sentencepiece as spm

# BPE
spm.SentencePieceTrainer.train(
    input='./CC-MAIN-20231128083443-20231128113443-00000.warc.wet',
    model_prefix='m_bpe',
    model_type='bpe',
    vocab_size=7000,
    user_defined_symbols=[])

# Unigram
spm.SentencePieceTrainer.train(
    input='./CC-MAIN-20231128083443-20231128113443-00000.warc.wet',
    model_prefix='m_unigram',
    model_type='unigram',
    vocab_size=7000,
    user_defined_symbols=[])
