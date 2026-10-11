---
status: accepted
---

# Each dataset owns one folder under `data/`, named by its dataset key

Dataset generation reads local files: source CSVs, torchvision and
Imagenette downloads, and feature cost files. They sat in `data/misc/`,
a name that said nothing, while feature costs sat apart in
`data/misc/feature_costs/<dataset key>.csv`. A feature cost is
dataset-defined (see `CONTEXT.md`), so splitting it from its dataset's
data meant a dataset's inputs lived in two places, and each new dataset
needed its own location chosen for it.

We decided that every file a dataset needs lives in `data/<dataset key>/`:
its source files under their original names, its feature costs in
`feature_costs.csv`, and anything else it brings, such as a license or
checksums. The folder exists even when it holds a single file. Synthetic
datasets have no source files, so their folder only exists when they have
feature costs. `load_feature_costs` finds the file from the dataset key
alone, through `DATA_DIR` in `src/afabench/core/types.py`, so a dataset needs
no config to have non-uniform feature costs.

## Considered options

- **Flat files directly in `data/`.** Rejected: a dataset's feature costs
  and extra files (license, checksums, schema) would need either a shared
  folder or a naming scheme, and the shared folder recreates the split.
- **`data/feature_costs/<dataset key>.csv` beside per-dataset files.**
  Rejected: it keeps one folder in `data/` that is not a dataset, and
  separates feature costs from the dataset that defines them.

## Consequences

- Files inside a dataset folder keep their upstream names
  (`bank-marketing.csv`, torchvision's `MNIST/`), so they still match what
  their source provides; the folder carries the dataset key.
- Most of `data/` is gitignored, so moving a dataset's folder means every
  checkout moves its local files by hand.
