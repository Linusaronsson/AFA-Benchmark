# CUBE-NM, noiseless (`cube_nm_without_noise`)

The [noiseless variant](../../../CONTEXT.md) of [CUBE-NM](cube_nm.md):
every noise standard deviation is 0, so perfect accuracy is attainable.
This page describes only what differs from CUBE-NM.

## Features

The context features are exactly one-hot, every block feature that is not
informative is exactly 0.5, and the three informative features of the
context's block are exactly the bits of the label, 0 or 1.

## Preprocessing in AFABench

As for CUBE-NM: no normalization, the context selection costs 1 and every
block feature 1, the pipeline's splits, and hard budgets 3, 5, 7, 10 and
14 (`all` profile).
