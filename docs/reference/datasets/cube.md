# CUBE (`cube`)

The synthetic [CUBE](../../../CONTEXT.md) dataset: each of 8 classes makes
three of 20 features informative, by setting them to the class's 3-bit
binary code, and leaves the rest noise. An instance is one draw, its label
the class.

Dataset generation draws the data anew for every dataset realization,
seeded by the realization index; there is no source file.

## References

- S. Zannone, J. M. Hernández-Lobato, C. Zhang and K. Palla. ODIN: optimal
  discovery of high-value information using model-based deep
  reinforcement learning. *ICML Workshop on Real-world Sequential Decision
  Making* (2019), Section 4.1: the variant AFABench's generator follows.
- T. Rückstieß, C. Osendorfer and P. van der Smagt. Minimizing data
  consumption with sequential online feature selection. *International
  Journal of Machine Learning and Cybernetics* 4, 235–243 (2013), and
  H. Shim, S. J. Hwang and E. Yang. Joint active feature acquisition and
  classification with variable-size set encoding. *NeurIPS* (2018): the
  CUBE datasets ODIN's is "inspired by".

## Instances and label

1,000 instances per dataset realization (`n_samples`), each label drawn
uniformly from the 8 classes, 0–7.

## Features

20 features, every one drawn from a normal distribution with mean 0.5 and
standard deviation 0.3 (`non_informative_feature_mean`,
`non_informative_feature_std`), except three. For an instance of class
*c*, features *c*, *c* + 1 and *c* + 2 instead hold the bits of *c*, least
significant first, each plus normal noise with standard deviation 0.3
(`informative_feature_std`):

| Class | Informative features | Their values |
| --- | --- | --- |
| 0 | 0, 1, 2 | 0, 0, 0 |
| 1 | 1, 2, 3 | 1, 0, 0 |
| 2 | 2, 3, 4 | 0, 1, 0 |
| 3 | 3, 4, 5 | 1, 1, 0 |
| 4 | 4, 5, 6 | 0, 0, 1 |
| 5 | 5, 6, 7 | 1, 0, 1 |
| 6 | 6, 7, 8 | 0, 1, 1 |
| 7 | 7, 8, 9 | 1, 1, 1 |

Features 0–9 are informative for some class; features 10–19 never are.

## Preprocessing in AFABench

- No normalization.
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 3, 5 and 10 (`all` profile).

## Variants

- [`cube_without_noise`](cube_without_noise.md): the noiseless variant.
- [`cube_nonuniform_costs`](cube_nonuniform_costs.md): two informative
  blocks at different costs.
- [`cube_nm`](cube_nm.md): a context feature selects the informative block.
