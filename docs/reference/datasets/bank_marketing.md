# Bank Marketing (`bank_marketing`)

Clients contacted by phone in a Portuguese bank's direct marketing
campaigns. An instance is one client, its features the client's data and
the campaign's contacts with them, and its label whether the client
subscribed a term deposit.

## Source

The prepared source is committed to the repository (ADR 0014) and checked
on every generation. `scripts/dataset_generation/prepare_source.py
bank_marketing` rebuilds it: it fetches the dataset through `ucimlrepo` and
writes the frame `ucimlrepo` returns, semicolon-separated, its label last.

| | |
| --- | --- |
| Provider | [UCI Machine Learning Repository, dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), "Bank Marketing", last updated 2023-08-18: its `bank-full` version, all 45,211 contacts with 16 inputs, ordered by date from May 2008 to November 2010 |
| Download | <https://archive.ics.uci.edu/static/public/222/data.csv>, through `ucimlrepo.fetch_ucirepo(id=222)` |
| File | `data/bank_marketing/bank-marketing.csv`: 45,211 clients, 16 features and `y` |
| SHA-256 | `7e3c9e305e7871ef398254dd01c1e3a1ff2b66d6c3e36666367cf07fdb0f8f03` |
| DOI | [10.24432/C5K306](https://doi.org/10.24432/C5K306) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), on UCI |

CC BY 4.0 requires attributing any use of the data; cite the papers below.
`data/bank_marketing/LICENSE.md` repeats this beside the file.

## References

- S. Moro, P. Cortez and P. Rita. A data-driven approach to predict the
  success of bank telemarketing. *Decision Support Systems* 62, 22–31
  (2014). [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
- S. Moro, P. Rita and P. Cortez. Bank Marketing [Dataset]. UCI Machine
  Learning Repository (2014).
  [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306).

## Instances and label

45,211 instances, one per client, in the source's order; none is dropped.

| Index | `y` | Instances |
| --- | --- | --- |
| 0 | `no` | 39,922 |
| 1 | `yes`: subscribed | 5,289 |

## Features

As UCI describes them, in feature order. The source leaves a categorical
value empty where UCI's documentation lists `unknown`.

| Index | Column | Meaning | Missing |
| --- | --- | --- | --- |
| 0 | `age` | Age, years | 0 |
| 1 | `job` | Type of job, one of 11 | 288 |
| 2 | `marital` | `divorced` (divorced or widowed), `married` or `single` | 0 |
| 3 | `education` | `primary`, `secondary` or `tertiary` | 1,857 |
| 4 | `default` | Has credit in default | 0 |
| 5 | `balance` | Average yearly balance, euros | 0 |
| 6 | `housing` | Has a housing loan | 0 |
| 7 | `loan` | Has a personal loan | 0 |
| 8 | `contact` | Contact type: `cellular` or `telephone` | 13,020 |
| 9 | `day_of_week` | Day of the month of the last contact, 1–31 (UCI's column name notwithstanding, #147) | 0 |
| 10 | `month` | Month of the last contact | 0 |
| 11 | `duration` | Duration of the last contact, seconds | 0 |
| 12 | `campaign` | Contacts with the client during this campaign, the last included | 0 |
| 13 | `pdays` | Days since the client was last contacted in a previous campaign; −1 if never, read as a number (#144) | 0 |
| 14 | `previous` | Contacts with the client before this campaign | 0 |
| 15 | `poutcome` | Outcome of the previous campaign: `failure`, `other` or `success` | 36,959 |

UCI notes that `duration` "highly affects the output target (e.g., if
duration=0 then y='no')" and is not known before a call is made, so it
"should be discarded if the intention is to have a realistic predictive
model". AFABench keeps it (#139).

## Preprocessing in AFABench

- Every categorical feature is label-encoded in alphabetical order of its
  values, a missing value being the value `nan`, a category of its own;
  the codes are then treated as numbers (#143).
- Every feature is z-normalized over the 45,211 instances before dataset
  generation splits them (#150).
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 2, 4 and 7 (`all` profile). Evaluation reports F1
  rather than accuracy for this dataset.
