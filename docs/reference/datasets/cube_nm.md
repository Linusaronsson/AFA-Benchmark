# CUBE-NM (`cube_nm`)

The synthetic [CUBE-NM](../../../CONTEXT.md) dataset, introduced by
AFABench: a context feature selects which of five CUBE-style blocks of 10
features is informative, so a method gains by acquiring the context first,
which a myopic method has no reason to do. An instance is one draw, its
label one of 8 classes.

Dataset generation draws the data anew for every dataset realization,
seeded by the realization index; there is no source file.

## References

- V. Schütz, H. Wu, R. Rezvan, L. Aronsson and M. Haghir Chehreghani.
  AFABench: a generic framework for benchmarking active feature
  acquisition. *KDD* (2026), arXiv:2508.14734, Section 4.3.
- [CUBE](cube.md), whose blocks these are.

## Instances and label

1,000 instances per dataset realization (`n_samples`), each drawn with a
context, 0–4, and a label, 0–7, both uniform and independent.

## Features

55 features: 5 context features, then 5 blocks of 10.

- Features 0–4 encode the context *k* one-hot, each plus normal noise with
  standard deviation 0.1 (`context_feature_std`).
- Block *k* (features 5 + 10*k* to 14 + 10*k*) is a CUBE block: for label
  *c*, its features *c*, *c* + 1 and *c* + 2 hold the bits of *c*, least
  significant first, each plus normal noise with standard deviation 0.3
  (`informative_feature_std`); see [CUBE](cube.md#features).
- Every other block feature is drawn from a normal distribution with mean
  0.5 and standard deviation 0.3 (`non_informative_feature_mean`,
  `non_informative_feature_std`), carrying no information.

A selection reveals either all five context features at once or one block
feature (`cube_nm` unmasker): 51 feature groups.

## Preprocessing in AFABench

- No normalization.
- Each context feature costs 1/5 (`use_cheap_context_features`), so the
  context selection costs 1; every block feature costs 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 3, 5, 7, 10 and 14 (`all` profile).

## Variants

- [`cube_nm_without_noise`](cube_nm_without_noise.md): the noiseless
  variant.
