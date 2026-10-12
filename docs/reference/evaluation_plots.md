# Evaluation performance plots

The visualization stage consumes probability-bearing
[plotting-ready evaluation tables](evaluation_dataframes.md#plotting-ready-tables).
Hard-budget and soft-budget comparisons are separate figure families. Metrics
are computed within each evaluation run, then summarized across seeds; see
[metric definitions](evaluation_dataframes.md#metrics-and-seed-summaries).

## Metric selection

Usual figures use `plotting.dataset_metrics`, a mapping from dataset key to
metric identifier, with `plotting.default_metric` for unlisted datasets.
Only these exact identifiers are supported:

| Identifier | Metric | Mean column | Standard deviation column |
| --- | --- | --- | --- |
| `accuracy` | Accuracy | `mean_accuracy` | `std_accuracy` |
| `f1` | Macro F1 | `mean_f_score` | `std_f_score` |
| `auroc` | AUROC | `mean_auroc` | `std_auroc` |

The Hydra defaults in `conf/scripts/plotting/common/default.yaml` are
`physionet_challenge_2012: f1`, `bank_marketing: f1`, and `default_metric: accuracy`.
`datasets_with_f_score` is replaced by this mapping, not supported alongside
it. The launcher accepts the same choices in a pipeline profile override:

```shell
workflow/run.py -n "plotting={dataset_metrics: {cube: auroc}, default_metric: accuracy}"
```

Quote the entire override so spaces and braces reach Snakemake intact.
The workflow forwards it as Hydra `++plotting.dataset_metrics={cube:auroc}`
and `plotting.default_metric=accuracy`. Dataset mappings merge into the
Hydra defaults; unmentioned datasets retain their metric choices. A direct
Hydra script invocation takes those Hydra overrides instead of the launcher
mapping. Other display settings remain in the existing Hydra display config.
There is no launcher metric flag.

## Optional per-dataset figures

The launcher flag `--per-dataset-plots` is off by default. Enabling it adds
figures rather than replacing usual figures. Each method set has its own
family, with one dataset per figure:

- Columns: Accuracy, Macro F1, AUROC.
- Rows: External classifier, Built-in classifier. The built-in row contains
  only methods with built-in results and is omitted when none exist.
- Metric panels have independent y scales and share x scales within a figure.
  Soft-budget views use accumulated cost; hard-budget and selection views
  retain their hard-budget and selection-count x variables.
- Existing method/style mappings, exclusions, largest-hard-budget trajectory
  selection and seed summaries apply. Unavailable AUROC is not replaced by
  zero or another metric.

The external usual-plot job (`plot_eval_perf_per_dataset`) owns the extra
output once per method set and reads the unsplit merged table, so built-in
results are included even though its usual figure uses external results.
The built-in usual job remains separate. Both declared output directories
must exist for this job to be cached; an existing usual plot does not satisfy
an additional target. Optional and usual output roots must not overlap,
including through symlinks. With the flag off, optional targets and their
rule are absent.

## Output layout

Below `<output_root>/plots/eval_split-<split>/initializer-<initializer>/`:

```text
eval_perf/
  method_set-<set>+classifier_type-<external|builtin>/
    dataset_set-<dataset-set>/...
eval_perf_per_dataset/
  method_set-<set>/
    dataset-<key>/
      hard_budget_normal.<format>
      hard_budget_traj.<format>
      soft_budget_2d_errors.<format>
      soft_budget_lines.<format>
```

Variants are present only when the corresponding setting has results.
The workflow writes PDF and SVG. The additional directories are independently
tracked Snakemake directory outputs; no classifier wildcard duplicates the
optional family. Allocation and walltime configuration use `plot_eval_perf`
for both usual-only and combined jobs.

## Script configuration

`scripts/plotting/plot_eval_perf.py` always generates usual figures from
`input`, `output_folder`, `dataset_sets` and the display config `plotting`.
Additional fields:

| Field | Default | Contract |
| --- | --- | --- |
| `per_dataset_plots` | `false` | Enable the additional family. |
| `per_dataset_input` | `null` | When enabled, required unsplit classifier-long merged table for the same method set, with `classifier`, `predicted_class` and full `predicted_probabilities` vectors. |
| `per_dataset_output_folder` | `null` | When enabled, required independent output root, neither equal to nor inside/containing the usual output root. |

Selecting a headline metric does not change the three metrics of the
additional family. Average precision (AP) is deferred to
[issue #130](https://github.com/MachineLearning-DecisionMaking/AFABench/issues/130).
