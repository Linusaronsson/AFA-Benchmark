# PhysioNet Challenge 2012 (`physionet_challenge_2012`)

ICU stays of the PhysioNet/Computing in Cardiology Challenge 2012. An
instance is one stay, its features the mean of each variable recorded in
the stay's first 48 hours, and its label whether the patient died in
hospital.

## Source

The prepared source is committed to the repository (ADR 0014) and checked
on every generation. `scripts/dataset_generation/prepare_source.py
physionet_challenge_2012` rebuilds it byte for byte from the provider's
files, checking each against its SHA-256. Downloading them took about 15
minutes in October 2026; the aggregation takes about 20 seconds.

| | |
| --- | --- |
| Provider | [PhysioNet](https://physionet.org/content/challenge-2012/1.0.0/), "Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012", version 1.0.0, published 2012-01-20 |
| Download | `set-a.tar.gz`, `set-b.tar.gz`, `set-c.tar.gz` (one record file per stay) and `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt`, from <https://physionet.org/files/challenge-2012/1.0.0/>; their SHA-256s are in `PhysionetChallenge2012Dataset.RAW_SHA256` |
| File | `data/physionet_challenge_2012/physionet.csv`: 12,000 stays, 41 features and `Outcome` |
| SHA-256 | `daca2ea78d228b93488f68938ab667a31e683adcbe916b77769633bc297aad91` |
| License | [Open Data Commons Attribution License v1.0](https://physionet.org/content/challenge-2012/view-license/1.0.0/) (ODC-By), on PhysioNet |

The ODC-By requires attributing any public use of the data. PhysioNet asks
to cite the Challenge paper and its standard PhysioNet citation, below.
`data/physionet_challenge_2012/LICENSE.md` repeats this beside the file.

## References

- I. Silva, G. Moody, D. J. Scott, L. A. Celi and R. G. Mark. Predicting
  in-hospital mortality of ICU patients: the PhysioNet/Computing in
  Cardiology Challenge 2012. *Computing in Cardiology* 39, 245–248 (2012).
- T. Pollard, B. E. Moody, L. Lehman, B. Gow, C. Fernandes, C. Xie,
  A. Johnson, R. G. Mark and T. Heldt. PhysioNet as a global platform for
  biomedical research. *Nature Health* (2026).
  [doi:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

## Instances and label

12,000 instances, one per stay: the 4,000 records of set A, then those of
sets B and C, each set in order of record file name. None is dropped.

The label is the outcome files' `In-hospital_death`:

| Index | `In-hospital_death` | Instances |
| --- | --- | --- |
| 0 | Survivor | 10,293 |
| 1 | Died in hospital | 1,707 |

## Features

Each record lists observations as (time, variable, value). Source
preparation takes, for each stay and variable, the mean of the variable's
values over the record, treating the Challenge's −1 as missing. A variable
the record never observes is missing for that stay. The six general
descriptors (record ID, age, gender, height, ICU type, weight) are recorded
once, on admission; the record ID is not a feature.

In feature order, with PhysioNet's units and the number of stays missing
each:

| Index | Column | Variable | Missing |
| --- | --- | --- | --- |
| 0 | `Age` | Age, years | 0 |
| 1 | `BUN` | Blood urea nitrogen, mg/dL | 183 |
| 2 | `Creatinine` | Serum creatinine, mg/dL | 183 |
| 3 | `GCS` | Glasgow Coma Score, 3–15 | 185 |
| 4 | `Gender` | 0 female, 1 male | 12 |
| 5 | `Glucose` | Serum glucose, mg/dL | 294 |
| 6 | `HCO3` | Serum bicarbonate, mmol/L | 207 |
| 7 | `HCT` | Hematocrit, % | 190 |
| 8 | `HR` | Heart rate, bpm | 184 |
| 9 | `Height` | Height, cm | 5,726 |
| 10 | `ICUType` | 1 coronary care, 2 cardiac surgery recovery, 3 medical, 4 surgical | 0 |
| 11 | `K` | Serum potassium, mEq/L | 249 |
| 12 | `Mg` | Serum magnesium, mmol/L | 292 |
| 13 | `NIDiasABP` | Non-invasive diastolic arterial blood pressure, mmHg | 1,531 |
| 14 | `NIMAP` | Non-invasive mean arterial blood pressure, mmHg | 1,536 |
| 15 | `NISysABP` | Non-invasive systolic arterial blood pressure, mmHg | 1,488 |
| 16 | `Na` | Serum sodium, mEq/L | 212 |
| 17 | `Platelets` | Platelets, cells/nL | 198 |
| 18 | `RespRate` | Respiration rate, bpm | 8,672 |
| 19 | `Temp` | Temperature, °C | 185 |
| 20 | `Urine` | Urine output, mL | 310 |
| 21 | `WBC` | White blood cell count, cells/nL | 213 |
| 22 | `Weight` | Weight, kg | 917 |
| 23 | `DiasABP` | Invasive diastolic arterial blood pressure, mmHg | 3,574 |
| 24 | `FiO2` | Fractional inspired O₂, 0–1 | 3,884 |
| 25 | `MAP` | Invasive mean arterial blood pressure, mmHg | 3,592 |
| 26 | `MechVent` | Mechanical ventilation, 0 false or 1 true | 4,419 |
| 27 | `PaCO2` | Partial pressure of arterial CO₂, mmHg | 2,945 |
| 28 | `PaO2` | Partial pressure of arterial O₂, mmHg | 2,946 |
| 29 | `SaO2` | O₂ saturation in hemoglobin, % | 6,632 |
| 30 | `SysABP` | Invasive systolic arterial blood pressure, mmHg | 3,573 |
| 31 | `pH` | Arterial pH, 0–14 | 2,891 |
| 32 | `ALP` | Alkaline phosphatase, IU/L | 6,903 |
| 33 | `ALT` | Alanine transaminase, IU/L | 6,787 |
| 34 | `AST` | Aspartate transaminase, IU/L | 6,786 |
| 35 | `Albumin` | Albumin, g/dL | 7,129 |
| 36 | `Bilirubin` | Bilirubin, mg/dL | 6,786 |
| 37 | `Lactate` | Lactate, mmol/L | 5,424 |
| 38 | `Cholesterol` | Cholesterol, mg/dL | 11,052 |
| 39 | `TroponinI` | Troponin-I, μg/L | 11,435 |
| 40 | `TroponinT` | Troponin-T, μg/L | 9,363 |

The order is that of the first stay's variables, sorted by name, followed
by each variable a later stay records first, in the order of the stays.
A few things follow from the aggregation and the records:

- `Weight` is both a general descriptor and a time series, often measured
  hourly; its mean covers both.
- `MechVent` is recorded only as 1, so every stay that has it has the mean
  1 and the feature says only whether ventilation was recorded.
- `Urine` is the mean of the recorded outputs, not their total.
- PhysioNet warns that "occasional outliers should be expected", and some
  means are implausible, such as a `pH` of 128.5, a `Temp` of 21.6 °C or a
  `Height` of 462 cm, and some blood pressures are 0. Source preparation
  keeps them (#148).
- Whether a mean suits each variable is open (#149).

## Preprocessing in AFABench

- A missing value is filled with its feature's mean over the 12,000
  instances, so whether a variable was measured is lost (#142); `MechVent`
  becomes constant and carries no information (#140).
- Every feature is then z-normalized over the 12,000 instances, before
  dataset generation splits them (#150). `ICUType`'s codes are treated as
  numbers (#143).
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 5, 10 and 15 (`all` profile). Evaluation reports F1
  rather than accuracy for this dataset.
