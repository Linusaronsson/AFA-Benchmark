---
status: accepted
---

# Every instance carries its generation index

When an evaluation shows something odd for one episode, we could not tell
which instance it was. Dataset generation shuffles and splits, and each
split bundle stored only the sliced features and labels, without the
positions they came from; only Imagenette kept indices, and its test split
indexed a different folder than train and val. The evaluation table's
`episode_id` is the loader order, which is not even the split index once
`eval_only_n_samples` samples a random subset.

We decided that every instance carries its **generation index** (see
`CONTEXT.md`) from dataset generation through to the evaluation table:

- **Datasets own it.** Providing generation indices is a required member of
  the `AFADataset` protocol, for every dataset, real or synthetic. Every
  dataset bundle saves them, loading a bundle without them raises, and
  `create_subset` composes them, so a subset of a subset stays correct.
- **Real-world datasets keep source order.** Dataset generation must not
  drop or reorder a real-world source's instances, or must keep the mapping
  itself, so that the generation index also locates the instance in the
  source, in every dataset realization. For a dataset loaded from several
  source parts (Imagenette's `train/` and `val/` folders), the generation
  index runs over the parts in a fixed order, so it is unique across splits.
- **Synthetic datasets** use generation order. Their generation index
  identifies an instance only together with its dataset realization.
- **Evaluation tables carry it per episode.** The evaluator writes
  `generation_index` and `split_index` columns into the raw table, and the
  transform step carries both through. This amends ADR 0002's evaluation
  identity columns, which are otherwise constant per file.

## Considered options

- **Indices in the bundle's free-form metadata, written by the generation
  scripts.** Rejected: the generator is the only place that knows them, any
  later `create_subset` loses them, and ADR 0002 records how free-form
  metadata drifted between the two generators.
- **Join evaluation rows to bundles on `episode_id`.** Rejected: episode
  order follows the loader, not the split, and results-only users have the
  tables but not the bundles.
- **Only the generation index in tables.** Rejected: the split index can be
  derived from it and the dataset bundle, but only with the bundle at hand,
  and it is cheap to write.
- **Upstream identifiers (file paths, source row ids).** Rejected as the
  general contract: they differ per source and do not exist for synthetic
  data. A dataset whose source needs one maps it from the generation index.
- **Optional indices, null when absent.** Rejected: there are no published
  outputs to stay compatible with, and a required member stops a new
  dataset from silently skipping it.
