# CUBE-NUC (`cube_nonuniform_costs`)

The [CUBE-NUC](../../../CONTEXT.md) variant of [CUBE](cube.md), with
non-uniform feature costs: the informative features appear in two blocks
of 10, the second costing twice as much, so a method that weighs costs
acquires from the first. An instance is one draw, its label one of 8
classes.

Dataset generation draws the data anew for every dataset realization,
seeded by the realization index; there is no source file.

## References

- [CUBE](cube.md)'s.

## Instances and label

1,000 instances per dataset realization (`n_samples`), each label drawn
uniformly from the 8 classes, 0–7.

## Features

20 features: two blocks of 10, each a CUBE block with its own noise. For
an instance of class *c*, features *c*, *c* + 1 and *c* + 2 of each block
(features *c* to *c* + 2 and 10 + *c* to 12 + *c*) hold the bits of *c*,
least significant first, each plus normal noise with standard deviation
0.3 (`informative_feature_std`); see [CUBE](cube.md#features). Every other
feature is drawn from a normal distribution with mean 0.5 and standard
deviation 0.3 (`non_informative_feature_mean`,
`non_informative_feature_std`). The second block takes the place of
CUBE's 10 features that are never informative.

## Preprocessing in AFABench

- No normalization.
- The second block's features cost twice the first's (`cost_scaling`),
  scaled so the mean feature cost is 1: 2/3 for each of features 0–9 and
  4/3 for each of features 10–19.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 2, 4 and 7 (`all` profile).
