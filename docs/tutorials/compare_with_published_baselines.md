# Comparing a new method with published baselines

This tutorial walks a **repository adopter** through the whole workflow: fork
AFABench, download a benchmark release's baseline results and shared
prerequisites, then train and evaluate only your own method and plot it
next to the published baselines. Published baselines are never retrained,
re-evaluated or re-transformed, and neither are the dataset instances,
external classifier or pretrained models you download.

If you only need the published tables and plots, without running AFABench,
see
[Finding and downloading results without AFABench](../release_publishing.md#finding-and-downloading-results-without-afabench)
instead. That is the **results-only journey**: every table is an ordinary
Parquet file readable with `pandas.read_parquet` (or any Parquet reader), and
every plot is an ordinary PDF or SVG file. Neither needs AFABench code to
load, and neither needs a Hugging Face account.

Only `scripts/release/snapshot.py download` talks to the release host.
Training, evaluation, transformation and plotting read and write local files
under `extra/output`. They need no Hugging Face account, token or service,
and work offline once the download is done.

## 1. Fork and add your method

Fork the repository and add your method as described in
[`add_method.md`](add_method.md): its scripts, its `method_options`
entry, its `soft_budget_params` entry and any registry entries. A
configured variant of an existing method also counts, for example a
published method with other hyperparameters. It needs only a new method
name in `method_options`:

```yaml
method_options:
  gdfs_low_lr:
    pretrained_model_name: "gdfs"   # reuse the published GDFS pretraining
    train_script_name: "gdfs"
    method_specific_params: ["lr=0.0005"]
    use_max_hard_budget_when_training_soft_budget: true
    eval_batch_size:
      default: 32
```

Give the method a **method name that no published method uses**, since
method names identify rows in the comparison tables and plots.

So that your method appears in plots, add it to the plotting display
configuration in `extra/conf/scripts/plotting/common/default.yaml`:

- `method_name_mapping`: its display name;
- `method_policy_family_mapping`: the colour family it is drawn in (an
  existing family, or a new one added to every entry of
  `method_family_color_schemes`).

`caption`, unset by default, is printed under every evaluation performance
plot. Set it when the comparison is not a scientific result, for example
when it uses a smoke-test release:

```yaml
caption: "Workflow demonstration from smoke runs, not scientific results"
```

## 2. Choose a release and read its provenance

Downloads default to the latest **full** release. Name a release id to pin
an older or a partial one (see
[Choosing the release](../release_publishing.md#choosing-the-release)).
Before relying on a release, read its provenance. `download` prints it,
and `extra/release_manifest.json` keeps it beside your outputs:

- `code.commit` and `code.dirty`: the AFABench revision that produced the
  release;
- `workflow_config`: the profile, config files, overrides and the merged
  configuration of the run;
- `settings`: initializer, evaluation split, dataset instances, Unmaskers,
  feature costs, classifiers and evaluation batch sizes;
- `scope` and `execution_mode`: a `test_only` or `smoke` release is never a
  benchmark result.

AFABench does not check that your checkout is compatible with the release's
commit. Compare the commit with your fork's history and read the release
notes ([`release_notes.md`](../release_notes.md)) for changes since that
commit that can affect results. A changed dataset, split, preprocessing,
classifier, acquisition semantics or metric can make your results
incomparable with the published ones even when every file still loads.

## 3. Download what the comparison needs

Download into the fork's `extra/output` (the default destination):

```shell
uv run python scripts/release/snapshot.py download --repo-id <repo_id> \
    --payload-category transformed_evaluation_table \
    --payload-category dataset_bundle \
    --payload-category classifier_bundle \
    --payload-category pretrained_model_bundle \
    --dataset cube --method random_dummy --method gdfs
```

| Payload category | Needed? | Why |
| --- | --- | --- |
| `transformed_evaluation_table` | Required | The baselines' plotting-ready tables, which the comparison aggregates. |
| `dataset_bundle` | Required | Your method is trained and evaluated on exactly the published dataset instances and splits. Without them, the workflow generates the datasets again. |
| `classifier_bundle` | Required | The external classifier makes the external-classifier predictions of every method on a dataset. Without it, the workflow trains a new one and your external-classifier results would come from a different classifier. |
| `pretrained_model_bundle` | Required if your method trains from a pretrained model the release has | Skips pretraining. The bundle comes with the time record its pretraining job wrote, which the workflow's time plot reads. |
| `raw_evaluation_table` | Optional | Acquisition histories, for your own analysis. The comparison does not read them. |
| `afa_method_bundle` | Optional | Only for re-evaluating a published method (other seeds, budgets, splits). Never needed to plot against published results. |
| `--output-category plot_results` | Optional | The published plots. |

The coverage options (`--dataset`, `--method`, `--budget-setting`, ...)
select the evaluations to compare with, and the bundle categories bring
what those evaluations were produced from. `--method` names published
methods, not yours. See
[Selecting what to download](../release_publishing.md#selecting-what-to-download).
Coverage the release lacks is reported; nothing is filled in from another
release.

### Existing files

`download` never overwrites silently. If any file it would write already
exists, including `extra/release_manifest.json`, it writes nothing and
lists the conflicts. `--overwrite` replaces exactly those files. So a
second download into the same fork, for example of more datasets, needs
`--overwrite`, if only for `release_manifest.json`. Do not download
another release into the same output tree: the restored manifest
describes one release only, and its outputs would be mixed with yours.

The workflow never writes to a restored reference table (section 4). It
does rebuild the merged tables and plots of every method set that contains
one of your methods. So if you downloaded `merged_results` or
`plot_results`, the files of those method sets are replaced by your
comparison, while method sets made only of published methods are left as
downloaded.

## 4. Configure the comparison

Start from the release's own workflow configuration, so that your run
expects the reference tables at exactly the eval split, initializer,
dataset instances and budgets the release has. Then layer your method on
top:

```shell
jq .workflow_config.merged extra/release_manifest.json > release_config.json
```

Remove cluster-specific keys such as `execution_site_file` from it if you
run elsewhere.

```yaml
# adopter.yaml
methods: [gdfs_low_lr]
reference_methods: [random_dummy, gdfs]
method_sets:
  my_comparison: [random_dummy, gdfs, gdfs_low_lr]
method_options:
  gdfs_low_lr: {...}   # as in section 1, unless it is in your config files
soft_budget_params:
  gdfs_low_lr:
    cube: []
```

- `methods` lists only the methods you produce locally.
- `reference_methods` lists the published methods to compare against.
  Their plotting-ready tables must be in `extra/output`. They join method
  sets and the evaluation performance aggregation, but no rule produces
  anything for them.
- `method_sets` names the groups plotted together. Sets without any of
  your methods are skipped, because the release already has their plots.

Snakemake merges config files in order, recursively, so `adopter.yaml`
replaces `methods` and adds to `method_options` and `soft_budget_params`.

## 5. Check the plan, then run

Ask Snakemake what it would run before running it:

```shell
uv run snakemake \
    --snakefile extra/workflow/snakefiles/orchestration/pipeline.smk \
    --configfile release_config.json adopter.yaml \
    --cores 4 --dry-run all
```

The plan must list only your method's jobs (`train_method`,
`eval_method`, `transform_eval_data`, its time record, plus `pretrain_model`
only if you did not download its pretrained model), and the comparison's
`merge_eval_perf`, `split_by_classifier_type`, `plot_eval_perf`,
`merge_time` and `plot_time`. If it lists `dataset_generation` or
`train_classifier`, a shared prerequisite is missing: download it rather
than letting the workflow produce a different one. Then run the same
command without `--dry-run`.

If a reference table is missing, the plan fails with Snakemake's
`MissingInputException`, naming the expected path. The reason can be a
budget, dataset instance or evaluation split your configuration asks for
but the release lacks, or a table you did not download. Download it, or
change your configuration to match the release. The workflow never
produces a reference table, so it never evaluates a published method in
its place.

The comparison ends up in:

```text
extra/output/merged_results/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-my_comparison+all.parquet
    method_set-my_comparison+classifier_type-builtin.parquet
    method_set-my_comparison+classifier_type-external.parquet
extra/output/plot_results/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-my_comparison+classifier_type-<builtin|external>/<dataset set>/
        hard_budget_normal.{pdf,svg}   hard_budget_traj.{pdf,svg}
        soft_budget_lines.{pdf,svg}    soft_budget_2d_errors.{pdf,svg}
```

## Why reference methods

Default aggregation enumerates every configured method's evaluations, and
Snakemake schedules the upstream work of each one. Putting published
baselines in `methods`, even with all their tables downloaded, therefore
retrains and re-evaluates them:

- the `all` target's time plot needs every method's `train_time.txt` and
  `eval_time.txt`, which only training and evaluation write;
- a restored table is also rebuilt whenever any prerequisite upstream of it
  is newer, or is missing and regenerated.

A reference method's plotting-ready tables are plain input files: the
transformation rule does not match reference methods, so Snakemake has no
rule to explore upstream of them. This is why the dry run lists none of
their jobs, and why a missing table fails rather than being produced.

## How rows stay comparable

- **No duplicate rows.** Each table is enumerated once, at its own path, and
  a method cannot be both in `methods` and in `reference_methods`.
- **Hard and soft budgets stay apart.** They are separate tables with
  separate `eval_hard_budget` and soft-budget-parameter columns, plotted in
  separate hard-budget and soft-budget plots.
- **Built-in and external classifiers stay apart.** Each row records its
  `classifier`, and the comparison is split by classifier type before
  plotting.
- **Other evaluation splits and initializers are never mixed in.** Tables
  are looked up under your configuration's `eval_split-*` and
  `initializer-*` folders. A release made with other settings fails the
  plan instead of being combined.

## Coverage limitations

- The comparison covers only evaluations the release has. Datasets,
  dataset instances, budgets or splits the release lacks fail the plan
  until you drop them from your configuration.
- Plotting-ready tables do not hold the dataset instance index or the
  evaluation split in their columns; the folders and the release
  manifest's `evaluation_tables` entries do.
- The time plot covers your methods only: published methods' time records
  are not downloaded.
- Whether the external classifier you evaluate with is the published one
  depends on downloading it. The dry run shows `train_classifier` when it is
  missing.
- Compatibility of your checkout with the release's commit is your
  responsibility (section 2).

## Tested

`test/workflow/test_adopter_journey.py` (marked `pipeline`) runs this
workflow on a smoke run. It publishes a test-only release of `random_dummy`
and `gdfs` on CUBE to a fake release host and downloads it into a
separate fork. It then adds a GDFS variant and checks the following:

- the dry-run plan and the executed jobs are only the variant's;
- the downloaded prerequisites are untouched;
- each baseline row appears once in the comparison;
- the captioned plots show all three methods;
- the downloaded tables and plots read with plain pandas and as plain files.

`test/workflow/test_reference_results.py` checks what the workflow plans
for reference methods, including missing tables and other evaluation splits
and initializers. Results from smoke releases are workflow demonstrations,
not scientific results.
