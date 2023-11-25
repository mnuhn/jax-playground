# Comparison of different training schemes for a simple NNs on MNIST

The intention of these experiments was to "do everything myself" (except the
gradient computation) when training a simple NN. 

You can start training with `python3 train.py`.

Here are some results I obtained using a dense network with `layers=[800,10]`.
Bias indicates whether a (learned) bias term is added (before the
non-linearity).

The matrices are initialized with gaussians $\mu=0$ and $\sigma=1/sqrt(N)$,
where $N$ is the "input" dimension. This is the Xavier initialization which
results in the output of the given layer to be Gaussian distributed with
$\mu=0$ and $\sigma=1$.

Here are some parameters and their results. I kept the network architecture
fixed and tried optimizers = [adam, sgd], batch size (BS) = [32,64,128],
learning rate (LR) = [0.001, 0.005, 0.01, 0.05]. I trained for a maximum of 300
epochs.

I report the best performance on the test set. Note that this is cheating! I
effectively tuned my parameters on the test set.

However, the purpose of this study was to see what influences the performance
of the training, and whether my self implemented SGD can actually achieve good
results (or whether there is something more to it).

I did run into numerical stability issues when I was implementing the loss
naively by first computing the output layer $p(c)$ (output of `softmax`) and
then computing the cross-entropy loss via $\sum_c \hat \delta(\hat c, c) \cdot
\log p(c)$. This can be simplified, by not computing the softmax (which
contains an `exp`), but directly compute the log-softmax. Without doing this,
training would diverge (especially using ADAM).

The most important finding for me is how much faster convergence with adam is,
and how robust the results are w.r.t. to learning rate.

Another (in hindsight obvious) finding: speedup with a GPU is really limited
with such small networks. I got maybe a ~3x speedup when using a GPU vs CPU for
a [800,10] network. When using [80000,10] however, I got a ~130x speedup.

Here are the results:

Optimizer               |  BS |    LR | Bias  | Best Acc. | Epochs
:-----------------------|----:|------:|------:|----------:|------:
adam                    |  32 | 0.001 | false |     98.59 |  57.78
adam                    |  64 | 0.001 |  true |     98.49 |  48.05
adam                    | 128 | 0.001 | false |     98.48 |  74.34
adam                    | 128 | 0.001 |  true |     98.47 |  53.31
adam                    |  32 | 0.001 |  true |     98.45 |  60.03
adam                    |  64 | 0.001 | false |     98.45 |  57.82
adam                    |  64 | 0.005 | false |     98.41 |  60.07
adam                    | 128 | 0.005 | false |     98.33 |  39.04
adam                    | 128 | 0.005 |  true |     98.22 |  54.81
adam                    |  64 | 0.005 |  true |     98.20 |  60.07
adam                    |  32 | 0.005 | false |     98.13 |  45.77
adam                    |  32 | 0.005 |  true |     98.13 |  39.77
adam                    | 128 |  0.01 | false |     98.11 |  69.08
adam                    | 128 |  0.01 |  true |     98.08 |  52.56
adam                    |  64 |  0.01 | false |     98.05 |  69.08
sgd                     |  32 |  0.05 |  true |     97.78 | 285.05
sgd, LR*=0.99/5 epochs  |  32 |  0.05 |  true |     97.76 | 273.05
adam                    |  64 |  0.01 |  true |     97.76 |  65.33
sgd, LR*=0.99/5 epochs  |  32 |  0.05 | false |     97.65 | 282.05
sgd, LR*=0.98/5 epochs  |  32 |  0.05 |  true |     97.64 | 294.05
sgd, LR*=0.99/10 epochs |  32 |  0.05 |  true |     97.64 | 264.04
sgd, LR*=0.99/10 epochs |  32 |  0.05 | false |     97.62 | 279.04


Here is a confusion matrix for the best result (on the test set):

```
label >      0    1    2    3    4    5    6    7    8    9
pred. v +--------------------------------------------------
   0    |  973    1    1    0    0    1    3    1    0    0
   1    |    0 1127    1    1    0    2    1    1    2    0
   2    |    2    6  999    5    1    0    4    3   11    1
   3    |    0    1    3  987    0   12    0    2    2    3
   4    |    1    1    3    1  962    0    5    3    2    4
   5    |    2    0    0    2    1  881    4    0    1    1
   6    |    2    4    0    1    2    2  946    0    1    0
   7    |    1    4    3    0    0    1    0 1013    3    3
   8    |    5    2    3    3    3    3    2    2  946    5
   9    |    4    3    0    1   10    7    0    1    3  980
```
