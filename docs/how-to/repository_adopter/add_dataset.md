# Adding a new dataset

## Overview

Datasets in AFA-Benchmark are serialized as **bundles** (directories containing the dataset data and metadata) and are deserialized by methods during training and evaluation. To add a new dataset to the benchmark, you need to define a dataset class, configure how it's generated, register it for deserialization, and integrate it with the pipeline and plotting system.

## Step-by-step guide

### 1. Define a dataset class

Define a dataset class in its own module, `src/afabench/datasets/<dataset_key>.py`, that implements the `AFADataset` protocol from `src/afabench/core/types.py`, and export it from `src/afabench/datasets/__init__.py`.

A dataset whose instances fit in memory as tensors subclasses
`InMemoryAFADataset` from `src/afabench/datasets/in_memory.py`. It supplies
its shapes, its config and how to build its tensors; the base owns the rest
of the protocol: subsets, indexing, and saving and loading `dataset.pt`.

**Minimal example:**

```python
from typing import ClassVar, final, override

import torch
import torch.nn.functional as F
from torch import Tensor

from afabench.datasets.in_memory import InMemoryAFADataset


@final
class MyDataset(InMemoryAFADataset):
    # The constructor arguments that generated the data; saved as the
    # bundle's config and restored as attributes on load
    config_keys: ClassVar[tuple[str, ...]] = ("n_samples",)

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([5])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([3])

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        super().__init__()

    @override
    def _build(self) -> tuple[Tensor, Tensor]:
        features = torch.randn(self.n_samples, 5)
        labels = F.one_hot(
            torch.randint(low=0, high=3, size=(self.n_samples,)),
            num_classes=3,
        ).float()
        return features, labels
```

The constructor sets the config attributes, then `super().__init__()` calls
`_build`, checks the tensors against `feature_shape` and `label_shape`, and
numbers the instances. A real-world tabular dataset read from one CSV file
subclasses `TabularCSVDataset` instead and declares its shapes, its file
argument and the file's SHA-256; see `DiabetesDataset`. A dataset that
cannot hold its instances in memory, like `ImagenetteDataset`, implements
the protocol itself.

A `TabularCSVDataset` reads its **prepared source**
(`docs/adr/0014-prepared-sources-are-committed.md`):

1. Implement `make_prepared_source`, which downloads the provider's raw
   files, checking each against a SHA-256 you pin, and returns the file's
   bytes; see `MiniBooNEDataset` or `PhysionetChallenge2012Dataset`.
2. Write the file with
   `uv run python scripts/dataset_generation/prepare_source.py <dataset key>`
   once its generation config exists (step 2), and set the class's `SHA256` to its SHA-256; the
   script refuses a file with another.
3. If the license permits redistributing the data, commit the file to
   `data/<dataset key>/` with a `LICENSE.md` giving its license and
   attribution. If it does not, or the provider states none, set
   `SOURCE_COMMITTED = False` and gitignore the folder instead; dataset
   generation then prepares the file when it is missing.

Every instance carries its **generation index**, its position in the
dataset your class produces, before dataset generation splits it
(`docs/adr/0004-instances-carry-their-generation-index.md`). Evaluation
tables record it per episode, so an odd result can be traced back to the
instance. The base numbers the instances `_build` returns `0..n-1`, composes
the indices in `create_subset`, and persists them: `load` raises
`MissingGenerationIndicesError` for a bundle saved without them. Your class
must **keep the source order for real-world data**: do not drop, sort or
shuffle the rows you read, so that generation index `i` is row `i` of the
source in every dataset realization. A `TabularCSVDataset` whose source
holds rows that are not instances may drop them in `_read_source`, keeping
the order and the frame index: the base saves that index as each instance's
**source row** (`get_source_rows`), as `MiniBooNEDataset` does. Any other
filtering or reordering must keep the mapping to the source itself.

`test/src/common/datasets/test_generation_indices.py` checks all of this for
every registered dataset class. Tell its `dataset_kwargs` how to build a
small instance of yours; if it reads a CSV, add it to `TABULAR_SOURCES` so
the source-order test covers it too.

The base also persists the **feature costs**
(`docs/adr/0011-feature-costs-stored-in-dataset-bundles.md`): `save` writes
`get_feature_acquisition_costs()` and `load` raises `MissingFeatureCostsError`
for a bundle saved without them. Dataset generation sets them before saving:

- **Unit costs** need nothing more.
- **Costs computed from config**, as in CUBE-NM: set `self.feature_costs` in
  the constructor, after `super().__init__()`.
- **Costs from a file**: put one value per feature, in flattened feature
  order, in `data/<dataset key>/feature_costs.csv`, then regenerate the
  dataset. `scripts/misc/generate_feature_costs.py` writes a random one. The
  file is read only at dataset generation, and a class that computes its
  own costs must not have one.

`test/src/common/datasets/test_feature_costs.py` checks that the costs reach
every split bundle, and `test/src/common/datasets/test_in_memory_dataset.py`
checks the base's subset and persistence rules once for every in-memory
dataset.

If your dataset is synthetic and should vary by dataset realization, make `accepts_seed()` return `True`, add a `seed` argument to `__init__` and list it in `config_keys`. The dataset generation script will pass one seed per dataset realization automatically.

### 2. Create an entry in dataset generation config

Create a config file in `conf/scripts/dataset_generation/dataset/` (e.g., `my_dataset.yaml`) to specify how the dataset should be generated:

```yaml
class_name: "MyDataset"
kwargs:
  n_samples: 10000
  # Add other constructor arguments as needed
```

The `class_name` must match the name of your dataset class in the registry, and `kwargs` are passed to the `__init__` method. The file name is the Hydra dataset key used by the pipeline, so the example above is selected with `dataset=my_dataset`.

### 3. Register the dataset class

Add your dataset class to the `REGISTERED_CLASSES` dictionary in `src/afabench/core/registry.py`:

```python
REGISTERED_CLASSES = {
    # ... existing entries ...
    "MyDataset": "afabench.datasets.my_dataset.MyDataset",
}
```

This allows the dataset generation script and methods to deserialize your dataset from bundles during training and evaluation via `afabench.core.registry.get_class()`.

### 4. Add to the Snakemake pipeline

List your dataset in one of the dataset configuration files in `workflow/conf/datasets/`. For example, in `workflow/conf/datasets/all.yaml`:

```yaml
datasets:
  - my_dataset
  # ... other datasets ...
```

The pipeline will generate `train.bundle`, `val.bundle`, and `test.bundle` under `output/production/datasets/my_dataset/{dataset_realization_index}/` for each selected dataset realization.

Every method then runs on it in the hard-budget setting, at the
`default` hard budgets of `workflow/conf/eval_hard_budgets/<variant>.yaml`
unless you give the dataset its own. No method runs it in the soft-budget
setting until the method lists it in `soft_budget_datasets` in
`workflow/conf/method_options/<variant>.yaml`
([ADR 0013](../../adr/0013-soft-budget-datasets-are-opt-in.md)); list it once
its soft-budget parameters suit it.

The scripts select the dataset through
`conf/components/dataset_key/my_dataset.yaml`; copy an existing one. Every
method then runs on its default hyperparameters, which suit tabular
datasets. Where a method needs other values on your dataset (an image
backbone, longer training), add a dataset override in
`conf/scripts/<stage>/<method>/dataset_override/my_dataset.yaml`; `imagenette`'s
files show the image case.

### 5. Record the dataset release policy

Add an entry for the dataset key in
`conf/release/dataset_redistribution.yaml`, including variants with their
own keys. Every pipeline dataset must have an entry: tests check coverage,
and release creation refuses encountered datasets without one. Maintain
this when adding the dataset or changing its terms, not for every release.

Record redistribution `status` as `permitted`, `restricted`, or explicitly
`unreviewed`, with the source, licence or terms, reviewer and relevant notes.
Do not infer permission from being able to generate a bundle.

Choose `bundle_policy` independently: `include` makes bundles eligible only
with a `permitted` review; `omit` always excludes them and requires an
`omission_reason`. For a synthetic dataset that users should regenerate,
use `omit` with a reason such as `synthetic dataset; regenerate locally`.
Copy an appropriate existing entry; never label a synthetic exclusion as a
redistribution restriction.
[Policy fields and what a review covers](../../reference/release_manifest.md#dataset-redistribution).

### 6. Add a readable name

(Optional but recommended) Add a display name to `dataset_name_mapping` in `conf/scripts/plotting/common/default.yaml`:

```yaml
dataset_name_mapping:
  # ... existing entries ...
  my_dataset: My Dataset Display Name
```

### 7. Add to dataset sets

(Optional) Add your dataset to one or more *dataset sets* in `dataset_sets` in `workflow/conf/dataset_sets/<variant>.yaml`. Dataset sets group datasets for organized plotting: each set with evaluation results gets its own `dataset_set-<name>/` plot folder.

```yaml
dataset_sets:
  set1:
    # ... existing datasets ...
    - my_dataset
```

Your dataset is plotted without this step: the `all` dataset set, which
holds every dataset of the run, always exists and cannot be configured.

### 8. Document the dataset

Write its page in `docs/reference/datasets/`, following the template in
[`docs/reference/datasets/README.md`](../../reference/datasets/README.md), and
list it there: where its data comes from and under which license, what its
instances, label and features are, and why it keeps only some of its
source's features, if it does.
