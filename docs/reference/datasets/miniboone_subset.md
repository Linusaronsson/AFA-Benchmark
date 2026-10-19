# MiniBooNE subset (`miniboone_subset`)

[MiniBooNE](miniboone.md) restricted to the 10 features of the
Learning-to-Measure (L2M) paper's MiniBooNE experiment. Its instances,
label and source are MiniBooNE's; this page describes only what differs.

## Source

MiniBooNE's: generation reads `data/miniboone/miniboone.csv` (ADR 0010),
with the same SHA-256 check, and
`scripts/dataset_generation/prepare_source.py miniboone_subset` prepares
the same file. Cite MiniBooNE's references and L2M below.

## References

- The features: Learning-to-Measure (L2M), arXiv:2510.12624v2, and its
  loader
  [`src/data/miniboone.py`](https://github.com/reAIM-Lab/Learning-To-Measure/blob/main/src/data/miniboone.py).

## Instances and label

MiniBooNE's 129,592 instances and their labels, with the same source rows.

## Features

In the source's order:

| Index | Column |
| --- | --- |
| 0 | `Feature 1` |
| 1 | `Feature 2` |
| 2 | `Feature 3` |
| 3 | `Feature 4` |
| 4 | `Feature 12` |
| 5 | `Feature 17` |
| 6 | `Feature 23` |
| 7 | `Feature 25` |
| 8 | `Feature 27` |
| 9 | `Feature 32` |

## Feature subset

AFABench keeps the 10 features L2M's loader keeps. The loader lists them in
the order `Feature 1`, `17`, `23`, `32`, `3`, `27`, `12`, `4`, `25`, `2`,
but drops the other columns from the file, so it too uses the source's
order. Neither the loader nor its config gives a rationale for the choice.

## Preprocessing in AFABench

As for MiniBooNE: every feature is z-normalized over the 129,592
instances, all feature costs are 1, and splits are the pipeline's. The
hard budgets are 1, 2, 3 and 4 (`all` profile); L2M trains one policy, at
hard budget 7, for every hard budget.

## Differences from L2M

- **Instances**: L2M's loader drops only rows with missing values, so it
  keeps the 472 placeholder and outlier events MiniBooNE drops.
- **Splits**: L2M's config (`configs/miniboone.yaml`, `data_seed: 123`)
  shuffles the events once, skips 1,000, takes the next 5,000 as the
  training pool and 10,000 of the rest as the test pool; AFABench uses its
  own splits per dataset realization.
- **Normalization**: L2M's loader keeps the source values and its
  `TabularDataset` normalizes each feature within each sampled task
  sequence; AFABench z-normalizes over all instances before splitting.
