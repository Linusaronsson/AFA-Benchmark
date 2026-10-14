# CUBE, noiseless (`cube_without_noise`)

The [noiseless variant](../../../CONTEXT.md) of [CUBE](cube.md): every
noise standard deviation is 0, so perfect accuracy is attainable. This
page describes only what differs from CUBE.

## Features

The 17 features that are not informative for an instance are all exactly
0.5, and its three informative features exactly the bits of its class, 0
or 1.

## Preprocessing in AFABench

As for CUBE: no normalization, feature costs 1, the pipeline's splits, and
hard budgets 3, 5 and 10 (`all` profile).
