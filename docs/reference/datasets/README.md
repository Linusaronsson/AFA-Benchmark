# Datasets

One page per [dataset key](../../../CONTEXT.md): where its data comes from,
what its instances, label and features are, and how AFABench prepares it.

Real-world tabular:

- [`actg`](actg.md): ACTG 175 HIV trial patients, failure event during
  follow-up
- [`bank_marketing`](bank_marketing.md): Bank Marketing clients, term
  deposit subscription
- [`ckd`](ckd.md): Chronic Kidney Disease patients
- [`diabetes`](diabetes.md): NHANES participants, fasting-glucose class
- [`metabric`](metabric.md): METABRIC breast tumours, molecular subtype
  from 11 genes
- [`miniboone`](miniboone.md): MiniBooNE events, electron or muon neutrino
- [`miniboone_subset`](miniboone_subset.md): MiniBooNE with L2M's 10
  features
- [`physionet_challenge_2012`](physionet_challenge_2012.md): ICU stays,
  in-hospital death

Real-world images:

- [`fashion_mnist`](fashion_mnist.md): Fashion-MNIST article images
- [`imagenette`](imagenette.md): Imagenette, 10 ImageNet classes
- [`mnist`](mnist.md): MNIST handwritten digits

Synthetic:

- [`cube`](cube.md): CUBE
- [`cube_without_noise`](cube_without_noise.md): CUBE, noiseless
- [`cube_nm`](cube_nm.md): CUBE-NM, a context feature selects the
  informative block
- [`cube_nm_without_noise`](cube_nm_without_noise.md): CUBE-NM, noiseless
- [`cube_nonuniform_costs`](cube_nonuniform_costs.md): CUBE-NUC, two
  informative blocks at different costs
- [`synthetic_mnist`](synthetic_mnist.md): Synthetic MNIST, class patterns
  in the right half of noise images
- [`synthetic_mnist_without_noise`](synthetic_mnist_without_noise.md):
  Synthetic MNIST, noiseless

## Page template

Name the page `<dataset key>.md` and add it to the list above. Leave out
a section that does not apply, such as Source for a synthetic dataset. A
variant of another dataset, such as a noiseless variant, gets a short page
that links to its base page and describes only what differs.

```md
# <Display name> (`<dataset key>`)

<One or two sentences: what an instance is and what the label is.>

## Source

Where the data comes from: the provider and version, the files source
preparation downloads, the prepared source under `data/<dataset key>/`
with its SHA-256, whether it is committed (ADR 0014), and the license with
the attribution it requires. Name the original provider when the download
is a mirror.

## References

The papers to cite for the data, and those it is taken from.

## Instances and label

How many instances there are, which source rows are dropped and why, the
classes in label order with their counts.

## Features

What each feature represents, in feature order, with its unit or scale.

## Feature subset

Only for a dataset that keeps some of its source's features: where the
subset comes from and its stated rationale, if any.

## Preprocessing in AFABench

What dataset generation does to the source values: encoding, missing
values, normalization, feature costs.

## Differences from <reference paper>

Only for a dataset taken from a paper's setup: what AFABench does
differently, such as splits, normalization or budgets.
```
