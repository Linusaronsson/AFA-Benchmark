# Diabetes (`diabetes`)

Participants of the US National Health and Nutrition Examination Survey
(NHANES), 1999–2016, as preprocessed by Kachuee et al. An instance is one
participant, its features 45 demographic, examination, laboratory and
questionnaire values, and its label a fasting-glucose class: normal,
prediabetes or diabetes. Most labels are probably imputed rather than
measured; see [Instances and label](#instances-and-label).

## Source

The source is committed to the repository (ADR 0014) and checked on every
generation. It is a copy of the GDFS repository's file; nothing in this
repository rebuilds it from NHANES, so it has no source preparation.

| | |
| --- | --- |
| File | `data/diabetes/diabetes.csv`: 92,062 participants, 45 columns named `0`–`44` and `Outcome` |
| SHA-256 | `0a7cb293c8819907a44f993e13e62fa00e93457ee7ccfab36ebe761b66df6c3b` |
| Copied from | `datasets/diabetes.csv` of [`iancovert/dynamic-selection`](https://github.com/iancovert/dynamic-selection) (the GDFS repository), byte-identical at commit `e2b6f74`; Ian Covert added it on 2023-01-02 |
| Preprocessing | Kachuee et al.'s `load_diabetes` in [`mkachuee/Opportunistic`](https://github.com/mkachuee/Opportunistic/blob/master/nhanes.py), from NHANES 1999–2016 |
| Original provider | [NHANES](https://www.cdc.gov/nchs/nhanes/), National Center for Health Statistics (NCHS), CDC |
| License | MIT for both repositories (Copyright 2023 Ian Covert; Copyright 2019 Mohammad Kachuee). NHANES public-use data are, in CDC's words, "generally" in the public domain, under the [NCHS data user agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html) |

The NCHS data user agreement binds every user of the data: use it only for
statistical reporting and analysis, make no attempt to identify anyone in
it, and do not link it with individually identifiable data. Cite the
papers below and NHANES in the
[style CDC suggests](https://wwwn.cdc.gov/nchs/nhanes/NhanesCitation.aspx).
`data/diabetes/LICENSE.md` repeats this beside the file.

## References

- M. Kachuee, K. Kärkkäinen, O. Goldstein, D. Zamanzadeh and
  M. Sarrafzadeh. Cost-sensitive diagnosis and learning leveraging public
  health data. arXiv:1902.07102 (2019).
- M. Kachuee, O. Goldstein, K. Kärkkäinen, S. Darabi and M. Sarrafzadeh.
  Opportunistic learning: budgeted cost-sensitive learning from data
  streams. *ICLR* (2019).
- I. Covert, W. Qiu, M. Lu, N. Kim, N. White and S.-I. Lee. Learning to
  maximize mutual information for dynamic feature selection. *ICML*
  (2023), arXiv:2301.00557, Appendix B.2: the GDFS paper, whose file this
  is.
- Centers for Disease Control and Prevention (CDC), National Center for
  Health Statistics (NCHS). National Health and Nutrition Examination
  Survey Data. Hyattsville, MD: U.S. Department of Health and Human
  Services, Centers for Disease Control and Prevention, 1999–2016.

## Instances and label

92,062 instances, one per row of the file; none is dropped. Kachuee's
loader shuffles the participants with an unseeded random permutation, and
the file holds no NHANES participant ID, so an instance cannot be traced
back to a participant.

The label, `Outcome`, is fasting plasma glucose (NHANES variable `LBXGLU`)
in three classes:

| Index | Class | Fasting glucose | Instances |
| --- | --- | --- | --- |
| 0 | Normal | at most 100 mg/dL | 16,156 |
| 1 | Prediabetes | between 100 and 125 mg/dL | 73,573 |
| 2 | Diabetes | at least 125 mg/dL | 2,333 |

Kachuee's loader fills a missing fasting glucose with the mean over all
participants before it thresholds it (`preproc_impute` on `LBXGLU`). Every
participant without a measurement therefore gets the class of the mean,
and the file does not record which participants those are. The 80% of
instances in class 1 suggest that the mean fell there and that most class 1
labels are imputed. Results on this dataset measure, in part, how well a
method predicts that a glucose measurement is missing (#137).

## Features

The file's columns are numbered, and neither repository documents which
column holds which variable. Kachuee's loader builds the features from
these NHANES variables, in this order; one-hot encoding expands the race
and smoking variables into several columns, for 45 in all, each acquired
on its own (#145):

- Demographics: gender (`RIAGENDR`), age (`RIDAGEYR`), race and ethnicity
  (`RIDRETH3` and `RIDRETH1`, one-hot), annual household income
  (`INDHHINC`), education (`DMDEDUC2`).
- Examination: four systolic and four diastolic blood pressure readings
  (`BPXSY1`–`BPXSY4`, `BPXDI1`–`BPXDI4`), body mass index (`BMXBMI`),
  waist circumference (`BMXWAIST`), height (`BMXHT`), upper leg length
  (`BMXLEG`), weight (`BMXWT`).
- Laboratory: total cholesterol (`LBXTC`), triglycerides (`LBXTR`),
  fibrinogen (`LBXFB`), LDL cholesterol (`LBDLDL`).
- Questionnaire: alcohol use (`ALQ101`, `ALQ120Q`), vigorous work and
  physical activity (`PAQ605`, `PAQ620`, `PAQ180`), sleep hours
  (`SLD010H`), smoking (`SMQ020`, one-hot; `SMD030`), blood relatives with
  diabetes (`MCQ250A`), history of high blood pressure (`BPQ020`).

Kachuee's loader fills a missing value of a numeric variable with its
mean and z-normalizes it; a one-hot column is 0 or 1, all 0 where the
variable is missing. The file has no missing value. Its columns 2–12 and
38–41 are the 15 one-hot columns, the others have mean 0 and standard
deviation 1, which fits the order above, but the file does not confirm it
column by column. Columns `40` and `41` are 1 for only 18 and 40
instances (#140).

## Preprocessing in AFABench

- Every feature is z-normalized over the 92,062 instances, before dataset
  generation splits them (#150); this rescales the one-hot columns and leaves the
  others nearly unchanged.
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 5, 10 and 15 (`all` profile, the default).
