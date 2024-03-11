Simple example of using SentencePiece, training it on some CommonCrawl data and
tokenizing some existing text.

# Unigram

See *Subword Regularization: Improving Neural Network Translation Models with
Multiple Subword Candidates* (https://arxiv.org/abs/1804.10959).

```
king -> ▁|king
ring -> ▁|ring
ringo -> ▁|ring|o
ringo starr -> ▁|ring|o|▁|s|tar|r
erring -> ▁|er|ring
erroring -> ▁|er|ro|ring
terrorizing -> ▁|ter|ro|ri|z|ing
doing -> ▁do|ing
wing -> ▁w|ing
ping -> ▁p|ing
inkling -> ▁|ink|ling
song -> ▁so|ng
bong -> ▁b|ong
nitting -> ▁|n|it|ting
ingo -> ▁in|go
bingo -> ▁b|ing|o
```

# bpe:

See *Neural Machine Translation of Rare Words with Subword Units*
(https://aclanthology.org/P16-1162/)

```
king -> ▁k|ing
ring -> ▁r|ing
ringo -> ▁r|ing|o
ringo starr -> ▁r|ing|o|▁st|ar|r
erring -> ▁er|r|ing
erroring -> ▁er|r|or|ing
terrorizing -> ▁t|er|r|or|iz|ing
doing -> ▁do|ing
wing -> ▁w|ing
ping -> ▁p|ing
inkling -> ▁in|kl|ing
song -> ▁s|ong
bong -> ▁b|ong
nitting -> ▁n|itt|ing
ingo -> ▁|ing|o
bingo -> ▁b|ing|o
```
