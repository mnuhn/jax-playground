# Implementation of LeNet for MNIST in JAX

This reaches ~ 98.9% accuracy (or 1.1% error rate) after 36 epochs.

```
python3 train.py

...

iteration=61886 correct=9892 total=10000 loss=7.621017061865132e-07
epoch=36.001 train_loss=0.000 test_loss=0.007:

label >      0    1    2    3    4    5    6    7    8    9
pred. v +--------------------------------------------------
   0    |  972    1    1    0    0    1    3    0    1    1
   1    |    0 1131    0    0    0    3    0    0    1    0
   2    |    0    0 1025    1    1    0    1    4    0    0
   3    |    0    0    1  986    0   15    0    3    2    3
   4    |    0    0    0    0  973    0    4    0    0    5
   5    |    1    0    0    4    0  884    1    0    1    1
   6    |    1    2    0    0    2    2  950    0    1    0
   7    |    0    6    4    1    1    0    0 1010    1    5
   8    |    2    0   11    6    2    7    0    4  938    4
   9    |    0    3    2    1    6    4    0    2    1  990
```
