import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='./CC-MAIN-20231128083443-20231128113443-00000.warc.wet',
    model_prefix='m',
    vocab_size=7000,
    user_defined_symbols=[])
