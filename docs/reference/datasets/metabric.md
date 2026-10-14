# METABRIC (`metabric`)

Primary breast tumours from the METABRIC cohort. An instance is one tumour,
its features the expression of 11 genes, and its label the tumour's
molecular subtype: one of the five PAM50 subtypes or claudin-low.

## Source

The source is committed to the repository unchanged (ADR 0014) and checked
on every generation, so that a generation index names the same tumour in
every checkout. `scripts/dataset_generation/prepare_source.py metabric`
downloads it again from Kaggle.

| | |
| --- | --- |
| Download | <https://www.kaggle.com/api/v1/datasets/download/raghadalharbi/breast-cancer-gene-expression-profiles-metabric>, a zip archive; no Kaggle account is needed |
| Kaggle page | [Breast Cancer Gene Expression Profiles (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) by Raghad Alharbi, version 1 (2020-05-26), the only one |
| File | `data/metabric/METABRIC_RNA_Mutation.csv`: 1,904 tumours, 693 columns |
| SHA-256 | `d4802594827667a71a75b5ae5c3cad1eb4346222052fcc792361c64abc7e43bc` |
| Original provider | [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric), study `brca_metabric`, "Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016)", from which the Kaggle author downloaded the clinical and genomic data |
| License | Open Database License (ODbL) for the database, Database Contents License (DbCL) for its contents, on Kaggle; cBioPortal publishes the study under the ODbL |

The ODbL requires attributing any public use of the data and sharing any
adapted database under the same license. Cite the METABRIC papers and
cBioPortal below. `data/metabric/LICENSE.md` repeats this beside the file.

The raw genotype and expression data of METABRIC are at the European
Genome-phenome Archive, study EGAS00000000083, under controlled access; the
processed data used here are public.

A different source file, such as a newer cBioPortal export, is not the same
dataset: generation rejects it. To use one, give it a dataset configuration
of its own in `conf/scripts/dataset_generation/dataset/`, under a new
dataset key, with its path as `root` and its SHA-256 as `sha256`.

## References

- C. Curtis et al. The genomic and transcriptomic architecture of 2,000
  breast tumours reveals novel subgroups. *Nature* 486, 346–352 (2012).
- B. Pereira et al. The somatic mutation profiles of 2,433 breast cancers
  refine their genomic and transcriptomic landscapes. *Nature
  Communications* 7, 11479 (2016).
- E. Cerami et al. The cBio Cancer Genomics Portal: an open platform for
  exploring multidimensional cancer genomics data. *Cancer Discovery* 2,
  401–404 (2012).
- J. Gao et al. Integrative analysis of complex cancer genomics and clinical
  profiles using the cBioPortal. *Science Signaling* 6, pl1 (2013).
- The features and label: Learning-to-Measure (L2M), arXiv:2510.12624v2,
  Appendix A.7.1, and its loader
  [`src/data/metabric.py`](https://github.com/reAIM-Lab/Learning-To-Measure/blob/main/src/data/metabric.py).

## Instances and label

1,898 instances: the 1,904 tumours of the file except the 6 whose subtype is
`NC` (not classified). No kept tumour has a missing value in the features or
the label. Each instance's source row is saved in the dataset bundles
(`get_source_rows`).

The label is the column `pam50_+_claudin-low_subtype`, in this order:

| Index | Source value | Subtype | Instances |
| --- | --- | --- | --- |
| 0 | `LumA` | Luminal A | 679 |
| 1 | `LumB` | Luminal B | 461 |
| 2 | `Her2` | HER2-enriched | 220 |
| 3 | `Basal` | Basal-like | 199 |
| 4 | `Normal` | Normal-like | 140 |
| 5 | `claudin-low` | Claudin-low | 199 |

## Features

Each feature is the mRNA expression z-score of one gene, measured on
Illumina HT-12 v3 microarrays: how many standard deviations the tumour's
expression lies from the gene's mean. In the file, every gene has mean 0 and
standard deviation 1 over the 1,904 tumours. In feature order:

| Index | Column | Gene |
| --- | --- | --- |
| 0 | `ccnb1` | Cyclin B1, which drives cells into mitosis; a proliferation marker |
| 1 | `cdk1` | Cyclin-dependent kinase 1, cyclin B1's partner kinase |
| 2 | `e2f2` | E2F transcription factor 2, an activator of cell-cycle genes |
| 3 | `e2f7` | E2F transcription factor 7, a repressor of E2F target genes |
| 4 | `stat5b` | STAT5B, a transcription factor of cytokine and prolactin signalling |
| 5 | `notch1` | The Notch 1 receptor |
| 6 | `rbpj` | RBPJ, the transcription factor through which Notch signals |
| 7 | `bcl2` | BCL2, an inhibitor of apoptosis |
| 8 | `egfr` | Epidermal growth factor receptor (HER1) |
| 9 | `erbb2` | HER2, the receptor amplified in HER2-positive tumours |
| 10 | `erbb3` | HER3, a HER2 dimerization partner |

The source orders the genes the same way.

## Feature subset

The file holds 31 clinical columns, z-scores for 489 genes and mutation
calls for 173 genes (the Kaggle page says 331 and 175). AFABench keeps the 11 genes that L2M v2 lists
as acquirable features for METABRIC. The paper gives no rationale for the
choice, and its loader none either. L2M v1 (arXiv:2510.12624v1, Appendix
A.5.1) also acquired ABCB1 and observed the age at diagnosis from the
start; v2 drops both, as does the published loader.

BCL2, CCNB1, EGFR and ERBB2 are among the 50 genes of the PAM50 classifier,
and the subtypes were assigned from the tumours' expression profiles, so the
label partly derives from the features rather than being independent of
them.

## Preprocessing in AFABench

- Every gene is z-normalized again over the 1,898 kept instances, as for
  every tabular CSV dataset, before dataset generation splits them; this
  changes the source values only slightly.
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test, for each
  dataset realization.
- The hard budgets are 3, 5 and 7 (`all` profile). No method runs METABRIC in
  the soft-budget setting; the L2M methods train one policy, at hard
  budget 7, for every hard budget.

## Differences from L2M v2

- **Splits**: L2M draws a 1,000-instance training pool and an 898-instance
  test pool once; AFABench uses its own splits per dataset realization.
- **Normalization**: L2M's loader keeps the source z-scores, then normalizes
  each feature within each sampled task sequence (Appendix A.7.3). Both use
  the source z-scores, which the whole cohort, test instances included,
  defined.
- **Missingness**: L2M evaluates under synthetic missingness of the
  retrospective data; AFABench's instances are complete.
- **Budgets**: L2M reports a sweep of acquisition budgets; AFABench
  evaluates hard budgets 3, 5 and 7.
- **Class order**: L2M's one-hot labels come from `pandas.get_dummies`, in
  the order Basal, Her2, LumA, LumB, Normal, claudin-low; per-class results
  must be matched by name.
