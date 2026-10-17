# Download published results

For a [results-only user](../../explanation/user_types.md#results-only-user):
download a benchmark release's tables and plots without installing
AFABench. Public releases need no Hugging Face account.

## 1. Choose a release

Releases are folders under `releases/` of the
[`afabench/results` Hugging Face dataset repository](https://huggingface.co/datasets/afabench/results/tree/main/releases).

Open a release folder and read its `README.md`. The generated summary lists
included datasets, methods, evaluation counts, splits, budgets and seeds,
with links to available tables and plots. It also includes any maintainer
release notes and dataset bundle exclusions. Choose a release id explicitly
whose coverage fits your needs; the summary lists do not imply that every
combination is present.

## 2. Find the tables you need

Follow the README's example-table links or output-folder links to browse
files. For exact evaluation identities and paths, open
`releases/<release_id>/release_manifest.json`. It records the
workflow configuration, the `coverage` of the release, and one
`evaluations` entry per evaluation with the path of its tables, the commit
that produced it, and its identity: method, dataset, dataset realization,
evaluation split, budgets and seeds.
[Fields](../../reference/release_manifest.md#fields).

## 3. Download tables and plots

Every file of a release is downloadable at

```text
https://huggingface.co/datasets/afabench/results/resolve/main/releases/<release_id>/output/<path>
```

with a browser, `curl`, or a Parquet reader that accepts URLs:

```python
import pandas as pd

df = pd.read_parquet(
    "https://huggingface.co/datasets/afabench/results/resolve/main/releases/"
    "<release_id>/output/eval_results_transformed/<table path>/eval_data.parquet"
)
```

Take `<path>` from the manifest:

- For accuracy and cost per number of selections, read the plotting-ready
  table (`transformed_path`).
- For acquisition histories, read the raw table (`raw_path`).
- For plots, browse `output/plots/`; they are PDF and SVG files.

Both tables carry their identity as columns. The
[columns of each table](../../reference/release_manifest.md#raw-versus-plotting-ready-tables).

To download many tables at once, selected by dataset, method or budget
setting, use the `download` command from a checkout of AFABench
([reference](../../reference/snapshot_command.md#download)).
