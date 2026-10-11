---
status: accepted
---

# Feature costs are fixed at dataset generation and stored in dataset bundles

Real-world datasets read their feature costs at runtime from
`data/<dataset key>/feature_costs.csv`, through a lookup on the `AFADataset`
protocol, and fell back to unit costs when the file was missing. `data/` is
gitignored, no feature costs file has ever been committed, and the scripts
that write one are run by hand. Training (AACO, DIME, GDFS, EDDI, the RL
methods) and evaluation (selection costs, hence forced stops under a hard
budget, and the acquisition cost under a soft budget) all repeated the
lookup. The same code and dataset bundles therefore gave different results
depending on a local file that no provenance record (ADR 0002) mentions, and
a fresh clone silently used unit costs. The lookup also inferred the
dataset key from the class name, so dataset keys that share a class
(`cube`, `cube_without_noise`) shared a file (#101).

We decided that dataset generation fixes the feature costs and every split
bundle stores them:

- **Generation resolves them once.** If the dataset's feature costs file
  exists, it sets the costs; otherwise the freshly constructed dataset does,
  either with costs it computes from its config (CUBE-NM, CUBE-NUC) or with
  unit costs. A dataset that computes its own costs and also has a file is
  an error, since one of them would be silently ignored.
- **Bundles carry them.** Each dataset's `save` writes them into its
  `dataset.pt`, so the bundle's content hash covers them, and `load` reads
  them back. Every later stage gets them from the loaded dataset; nothing
  reads the feature costs file after generation.
- **No backfill.** Loading a dataset bundle saved without feature costs
  raises `MissingFeatureCostsError`; such bundles are regenerated, as for
  generation indices (ADR 0004).
- **Unit costs stay the default** (`CONTEXT.md`): a dataset without a file
  gets unit costs, but they are now recorded in its bundles like any other
  costs.

An evaluation's feature costs are read by following its record's
`eval_dataset` input to the dataset bundle, as ADR 0002 intends for facts
that are not identity columns.

## Considered options

- **Commit the feature costs files used for published results.** Rejected:
  it fixes fresh clones, but the costs stay separate from the bundles they
  apply to, so a changed file still changes results of unchanged bundles
  without any record saying so.
- **Costs in the bundle's free-form metadata, or a separate file in
  `data/`.** Rejected: metadata is JSON and drifts between generators
  (ADR 0002), and a separate file would need every dataset's `save` and
  `load` to agree on it anyway; `dataset.pt` already holds the dataset's
  tensors.
- **Fall back to the file, then unit costs, for bundles without costs.**
  Rejected: it keeps the very dependency on a local file this decision
  removes.
- **A missing file is an error for real-world datasets.** Rejected: unit
  cost is the documented default, and recording it in the bundle already
  makes the choice visible.

## Consequences

- This amends ADR 0010, which had `load_feature_costs` find the file at
  runtime from the dataset key. The file stays where ADR 0010 puts it, but
  is read only by dataset generation, from the selected dataset config's
  key.
- Changing a feature costs file has no effect until the dataset is
  regenerated, and regenerating changes the bundles' content hashes.
- Dataset bundles generated before this decision no longer load.
- Methods that train on image datasets without consulting the dataset's
  costs (DIME's image training uses unit costs) are unchanged; they are as
  reproducible as before, but ignore non-unit costs.
