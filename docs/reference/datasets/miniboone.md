# MiniBooNE (`miniboone`)

Events of the MiniBooNE neutrino experiment. An instance is one event, its
features 50 particle identification variables, and its label whether the
event is signal (an electron neutrino) or background (a muon neutrino).

## Source

The prepared source is committed to the repository (ADR 0014) and checked
on every generation. `scripts/dataset_generation/prepare_source.py
miniboone` rebuilds it byte for byte from the provider's archive: it reads
`MiniBooNE_PID.txt`, whose first line counts the signal and the background
events that follow in that order, and writes the events as a CSV with
columns `Feature 1`–`Feature 50` and `Outcome`, 1 for signal.

| | |
| --- | --- |
| Provider | [UCI Machine Learning Repository, dataset 199](https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification), "MiniBooNE particle identification", donated on 2010-12-12 |
| Download | <https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip> (SHA-256 `af4573ca81797ca1c5bfc623f0fdf26ed07e626e2717c322224f96e8795f8752`), holding `MiniBooNE_PID.txt` |
| File | `data/miniboone/miniboone.csv`: 130,064 events, 50 features and `Outcome` |
| SHA-256 | `e8f5a61b08ac12304086a0326603dabb3400d57c691d4630c37e69492effb0ea` |
| Copied from | `datasets/miniboone.csv` of [`iancovert/dynamic-selection`](https://github.com/iancovert/dynamic-selection) (the GDFS repository), which is the same file |
| DOI | [10.24432/C5QC87](https://doi.org/10.24432/C5QC87) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), on UCI |

CC BY 4.0 requires attributing any use of the data; cite the papers below.
`data/miniboone/LICENSE.md` repeats this beside the file.

## References

- B. P. Roe, H.-J. Yang, J. Zhu, Y. Liu, I. Stancu and G. McGregor.
  Boosted decision trees as an alternative to artificial neural networks
  for particle identification. *Nuclear Instruments and Methods in Physics
  Research A* 543, 577–584 (2005).
- B. Roe. MiniBooNE particle identification [Dataset]. UCI Machine
  Learning Repository (2005).
  [doi:10.24432/C5QC87](https://doi.org/10.24432/C5QC87).

## Instances and label

129,592 instances: the file's 130,064 events in its order, except

- the 468 events whose every feature is −999, an undocumented placeholder
  (457 background, 11 signal), and
- the 4 background events whose `Feature 20` is −17,256.7, −12,790,
  −9,670.65 or 16,000,900, against −5.2 to 7.5 for every other event.

Each instance's source row is saved in the dataset bundles
(`get_source_rows`).

| Index | `Outcome` | Event | Instances |
| --- | --- | --- | --- |
| 0 | 0 | Background: muon neutrino | 93,104 |
| 1 | 1 | Signal: electron neutrino | 36,488 |

## Features

`Feature 1`–`Feature 50`, in the provider's order: the 50 particle
identification variables of each event. UCI does not say what each
variable measures, and none has a unit. No value is missing.

## Preprocessing in AFABench

- Every feature is z-normalized over the 129,592 instances, before dataset
  generation splits them.
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 5, 10 and 15 (`all` profile).
