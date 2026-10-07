# Download published results

For a [results-only user](../explanation/user_types.md#results-only-user):
download a benchmark release's tables and plots without installing
AFABench. Public releases need no Hugging Face account.

## 1. Choose a release

Releases are folders under `releases/` of the AFABench Hugging Face dataset
repository:

```text
https://huggingface.co/datasets/<repo_id>/tree/main/releases
```

Prefer the newest release with scope `full`; a `partial` release covers
only some datasets, methods or settings. Each release's entry in the
[release notes](../reference/release_notes.md) says what changed since the
previous one.

## 2. Read the release manifest

Open `releases/<release_id>/release_manifest.json`. It records the
producing commit, the workflow configuration, the `coverage` of the
release, and one `evaluation_tables` entry per evaluation with the path of
its tables and its identity: method, dataset, dataset instance, evaluation
split, budgets and seeds. [Fields](../reference/release_manifest.md#fields).

## 3. Download tables and plots

Every file of a release is downloadable at

```text
https://huggingface.co/datasets/<repo_id>/resolve/main/releases/<release_id>/output/<path>
```

with a browser, `curl`, or a Parquet reader that accepts URLs:

```python
import pandas as pd

df = pd.read_parquet(
    "https://huggingface.co/datasets/<repo_id>/resolve/main/releases/"
    "<release_id>/output/eval_results_transformed/<table path>/eval_data.parquet"
)
```

Take `<path>` from the manifest:

- For accuracy and cost per number of selections, read the plotting-ready
  table (`transformed_path`).
- For acquisition histories, read the raw table (`raw_path`).
- For plots, browse `output/plot_results/`; they are PDF and SVG files.

Plotting-ready tables have no dataset instance or evaluation split
column; take both from the table's `evaluation_tables` entry. The
[columns of each table](../reference/release_manifest.md#raw-versus-plotting-ready-tables).

To download many tables at once, selected by dataset, method or budget
setting, use the `download` command from a checkout of AFABench
([reference](../reference/snapshot_command.md#download)).
