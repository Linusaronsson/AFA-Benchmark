# Download published results

For a [results-only user](../../explanation/user_types.md#results-only-user):
download tables and plots without installing AFABench or creating an account.

## 1. Choose a release

Open the [published releases](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and read a release's README. Check coverage, caveats and exclusions before
choosing its release id.

## 2. Find the files

Follow the README's table and folder links. For exact evaluation combinations,
open `releases/<release_id>/release_manifest.json` and inspect `evaluations`:

- Use `transformed_path` for plotting-ready prediction and cost tables.
- Use `raw_path` for acquisition histories.
- Browse `output/plots/` for PDF and SVG figures.

[Manifest fields and table columns](../../reference/release_manifest.md).

## 3. Download the selected files

Use a browser, `curl`, or a Parquet reader with the release id and path:

```text
https://huggingface.co/datasets/afabench/results/resolve/main/releases/<release_id>/output/<path>
```

For example:

```python
import pandas as pd

df = pd.read_parquet(
    "https://huggingface.co/datasets/afabench/results/resolve/main/releases/"
    "<release_id>/output/eval_results_transformed/<table path>/eval_data.parquet"
)
```

For bulk downloads from an AFABench checkout, use the
[`download` command](../../reference/snapshot_command.md#download).
