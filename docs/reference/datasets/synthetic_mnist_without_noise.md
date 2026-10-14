# Synthetic MNIST, noiseless (`synthetic_mnist_without_noise`)

The [noiseless variant](../../../CONTEXT.md) of
[Synthetic MNIST](synthetic_mnist.md): the noise standard deviation is 0,
so perfect accuracy is attainable. This page describes only what differs
from Synthetic MNIST.

## Features

Every pixel outside the class's pattern is 0, and every pixel of the
pattern 0.8. All instances of a class are the same image.

## Preprocessing in AFABench

As for Synthetic MNIST: feature costs 1, 4 × 4 patches as feature groups,
the pipeline's splits, and hard budgets 5, 10 and 15 (`all` profile).
