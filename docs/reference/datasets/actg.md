# ACTG 175 (`actg`)

Adults with HIV from the AIDS Clinical Trials Group Study 175 trial. An
instance is one patient, its features baseline and follow-up measurements,
and its label whether the patient had a failure event (`cid`) during
follow-up.

## Source

Generation prepares the source when it is missing: it fetches the dataset
through `ucimlrepo` and writes the frame `ucimlrepo` returns, its label
last. It checks the file on every generation. The source is not committed
to the repository: UCI states no license for this dataset (see License).

| | |
| --- | --- |
| Provider | [UCI Machine Learning Repository, dataset 890](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175), "AIDS Clinical Trials Group Study 175", linked on 2023-09-25 and last updated 2023-11-03 |
| Download | <https://archive.ics.uci.edu/static/public/890/data.csv>, through `ucimlrepo.fetch_ucirepo(id=890)` |
| File | `data/actg/actg.csv`: 2,139 patients, 23 features and `target` |
| SHA-256 | `71493efd4ac3d8edf9edb43df83f0cd91b4cfe73bc9f96278fa8b536e8131cda` |
| DOI | [10.24432/C5ZG8F](https://doi.org/10.24432/C5ZG8F) |
| License | None stated. UCI says "See linked dataset for licensing information" and links the trial's [ClinicalTrials.gov record NCT00000625](https://clinicaltrials.gov/study/NCT00000625), which gives no data license |

UCI asks users to "follow the acknowledgment policy on the original
dataset website"; cite the trial paper and the UCI dataset below.

## References

- S. M. Hammer et al. A trial comparing nucleoside monotherapy with
  combination therapy in HIV-infected adults with CD4 cell counts from 200
  to 500 per cubic millimeter. *New England Journal of Medicine* 335,
  1081–1090 (1996).
- S. Hammer et al. AIDS Clinical Trials Group Study 175 [Dataset]. UCI
  Machine Learning Repository (1996).
  [doi:10.24432/C5ZG8F](https://doi.org/10.24432/C5ZG8F).

## Instances and label

2,139 instances, one per patient, in the source's order; none is dropped.
UCI's patient ID (`pidnum`) is not a feature.

| Index | `cid` | Meaning | Instances |
| --- | --- | --- | --- |
| 0 | 0 | Censored: no failure event observed | 1,618 |
| 1 | 1 | Failure event | 521 |

## Features

As UCI describes them, in feature order:

| Index | Column | Meaning |
| --- | --- | --- |
| 0 | `time` | Days to the failure event or censoring |
| 1 | `trt` | Treatment arm: 0 ZDV only, 1 ZDV + ddI, 2 ZDV + Zal, 3 ddI only |
| 2 | `age` | Age at baseline, years |
| 3 | `wtkg` | Weight at baseline, kg |
| 4 | `hemo` | Hemophilia (0 no, 1 yes) |
| 5 | `homo` | Homosexual activity (0 no, 1 yes) |
| 6 | `drugs` | History of IV drug use (0 no, 1 yes) |
| 7 | `karnof` | Karnofsky score, 0–100 (70–100 in the data) |
| 8 | `oprior` | Non-ZDV antiretroviral therapy before the trial (0 no, 1 yes) |
| 9 | `z30` | ZDV in the 30 days before the trial (0 no, 1 yes) |
| 10 | `zprior` | ZDV before the trial (0 no, 1 yes); 1 for every patient |
| 11 | `preanti` | Days of antiretroviral therapy before the trial |
| 12 | `race` | 0 white, 1 non-white |
| 13 | `gender` | 0 female, 1 male |
| 14 | `str2` | Antiretroviral history: 0 naive, 1 experienced |
| 15 | `strat` | Antiretroviral history: 1 naive, 2 more than 0 and at most 52 weeks, 3 more than 52 weeks |
| 16 | `symptom` | 0 asymptomatic, 1 symptomatic |
| 17 | `treat` | 0 ZDV only, 1 other arms |
| 18 | `offtrt` | Off treatment before week 96 ± 5 (0 no, 1 yes) |
| 19 | `cd40` | CD4 count at baseline |
| 20 | `cd420` | CD4 count at week 20 ± 5 |
| 21 | `cd80` | CD8 count at baseline |
| 22 | `cd820` | CD8 count at week 20 ± 5 |

`time` and `offtrt` are recorded during follow-up, so they carry
information about the outcome: a failure event ends a patient's follow-up
time (#138). `zprior` is constant, so it carries no information (#140).
`str2` is exactly `strat` > 1 and `treat` exactly `trt` > 0 (#141).

## Preprocessing in AFABench

- No value is missing. Every feature is z-normalized over the 2,139
  instances before dataset generation splits them (#150); `zprior` becomes
  0. The integer codes of `trt` and `strat` are treated as numbers (#143).
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 3, 5 and 10 (`all` profile).
