# Datasets

One page per [dataset key](../../../CONTEXT.md): where its data comes from,
what its instances, label and features are, and how AFABench prepares it.
Dataset keys without a page are not documented yet (#133).

None yet.

## Page template

Name the page `<dataset key>.md` and add it to the list above. Leave out
a section that does not apply, such as Source for a synthetic dataset.

```md
# <Display name> (`<dataset key>`)

<One or two sentences: what an instance is and what the label is.>

## Source

Where dataset generation gets the data: the URL it downloads from or the
file the user places, the file name under `data/<dataset key>/`, the
version and its SHA-256, and the license with the attribution it requires.
Name the original provider when the download is a mirror.

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
