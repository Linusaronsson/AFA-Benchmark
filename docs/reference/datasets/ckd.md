# Chronic Kidney Disease (`ckd`)

Patients of one hospital, collected over nearly two months (UCI). An
instance is one patient, its features blood and urine tests and medical history, and
its label whether the patient has chronic kidney disease.

## Source

The prepared source is committed to the repository (ADR 0014) and checked
on every generation. `scripts/dataset_generation/prepare_source.py ckd`
rebuilds it: it fetches the dataset through `ucimlrepo` and writes the
frame `ucimlrepo` returns, the label last.

| | |
| --- | --- |
| Provider | [UCI Machine Learning Repository, dataset 336](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease), "Chronic Kidney Disease", donated on 2015-07-02 and last updated 2024-03-04 |
| Download | <https://archive.ics.uci.edu/static/public/336/data.csv>, through `ucimlrepo.fetch_ucirepo(id=336)` |
| File | `data/ckd/chronic_kidney_disease.csv`: 400 patients, 24 features and `target` |
| SHA-256 | `45fb1b3b76b1a1a0f3bf1eab70796c76137ecc922e0f7c4c338141ae6e258f5b` |
| DOI | [10.24432/C5G020](https://doi.org/10.24432/C5G020) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), on UCI |

CC BY 4.0 requires attributing any use of the data; cite the dataset
below. `data/ckd/LICENSE.md` repeats this beside the file.

## References

- L. Rubini, P. Soundarapandian and P. Eswaran. Chronic Kidney Disease
  [Dataset]. UCI Machine Learning Repository (2015).
  [doi:10.24432/C5G020](https://doi.org/10.24432/C5G020).

## Instances and label

400 instances, one per patient, in the source's order; none is dropped.
The label is UCI's `class`, stripped of surrounding whitespace:

| Index | `class` | Instances |
| --- | --- | --- |
| 0 | `notckd` | 150 |
| 1 | `ckd` | 250 |

## Features

As UCI describes them, in feature order, with the number of patients
missing each:

| Index | Column | Meaning | Missing |
| --- | --- | --- | --- |
| 0 | `age` | Age, years | 9 |
| 1 | `bp` | Blood pressure, mm Hg | 12 |
| 2 | `sg` | Urine specific gravity: 1.005, 1.010, 1.015, 1.020 or 1.025 | 47 |
| 3 | `al` | Urine albumin, 0–5 | 46 |
| 4 | `su` | Urine sugar, 0–5 | 49 |
| 5 | `rbc` | Red blood cells in urine: `abnormal` or `normal` | 152 |
| 6 | `pc` | Pus cells: `abnormal` or `normal` | 65 |
| 7 | `pcc` | Pus cell clumps: `notpresent` or `present` | 4 |
| 8 | `ba` | Bacteria: `notpresent` or `present` | 4 |
| 9 | `bgr` | Random blood glucose, mg/dL | 44 |
| 10 | `bu` | Blood urea, mg/dL | 19 |
| 11 | `sc` | Serum creatinine, mg/dL | 17 |
| 12 | `sod` | Sodium, mEq/L | 87 |
| 13 | `pot` | Potassium, mEq/L | 88 |
| 14 | `hemo` | Hemoglobin, g | 52 |
| 15 | `pcv` | Packed cell volume | 71 |
| 16 | `wbcc` | White blood cell count, cells/mm³ | 106 |
| 17 | `rbcc` | Red blood cell count, millions/mm³ | 131 |
| 18 | `htn` | Hypertension: `no` or `yes` | 2 |
| 19 | `dm` | Diabetes mellitus: `no` or `yes`, and once `\tno` | 2 |
| 20 | `cad` | Coronary artery disease: `no` or `yes` | 2 |
| 21 | `appet` | Appetite: `good` or `poor` | 1 |
| 22 | `pe` | Pedal edema: `no` or `yes` | 1 |
| 23 | `ane` | Anemia: `no` or `yes` | 1 |

## Preprocessing in AFABench

- Every categorical feature is label-encoded in alphabetical order of its
  values. The source's one `\tno` in `dm` is a value of its own, so `dm`
  has three categories (#146).
- A missing value is filled with its feature's mean over the 400
  instances, after encoding, so a categorical feature can take a value
  between its categories (#142).
- Every feature is then z-normalized over the 400 instances, before
  dataset generation splits them (#150).
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 2, 4 and 7 (`all` profile).
