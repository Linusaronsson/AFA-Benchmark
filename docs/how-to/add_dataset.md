# Adding a new dataset

## Overview

Datasets in AFA-Benchmark are serialized as **bundles** (directories containing the dataset data and metadata) and are deserialized by methods during training and evaluation. To add a new dataset to the benchmark, you need to define a dataset class, configure how it's generated, register it for deserialization, and integrate it with the pipeline and plotting system.

## Step-by-step guide

### 1. Define a dataset class

Define a dataset class in `afabench/datasets/datasets.py` that implements the `AFADataset` protocol from `afabench/core/types.py`.

**Minimal example:**


```python
from collections.abc import Sequence
from pathlib import Path
from typing import Self, override

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from afabench.core.types import AFADataset, GenerationIndices
from afabench.datasets.utils import (
    default_create_subset,
    load_generation_indices,
)


class MyDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
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

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @override
    def get_generation_indices(self) -> GenerationIndices:
        return self.generation_indices

    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples = n_samples

        self.features = torch.randn(n_samples, 5)
        self.labels = F.one_hot(
            torch.randint(low=0, high=3, size=(self.n_samples,)),
            num_classes=3,
        ).float()
        self.generation_indices = torch.arange(n_samples)

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Tensor, Tensor]:
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "generation_indices": self.generation_indices,
                "config": {
                    "n_samples": self.n_samples,
                },
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.n_samples = data["config"]["n_samples"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.generation_indices = load_generation_indices(
            data, path / "dataset.pt", len(obj.features)
        )
        return obj
```

Every instance carries its **generation index**, its position in the
dataset your constructor produces, before dataset generation splits it
(`docs/adr/0004-instances-carry-their-generation-index.md`). Evaluation
tables record it per episode, so an odd result can be traced back to the
instance. Your class must:

- **Number its instances in generation order.** The constructor sets
  `generation_indices` to `0..n-1`; the generation script checks this before
  splitting.
- **Keep the source order for real-world data.** Do not drop, sort or shuffle
  the rows you read, so that generation index `i` is row `i` of the source in
  every dataset realization. If a source must be filtered or reordered, keep
  the mapping to the source yourself.
- **Compose them in `create_subset`.** A subset's generation indices are the
  parent's at the selected positions. `default_create_subset` does this for
  in-memory datasets with `features`, `labels` and `generation_indices`.
- **Persist them.** `save` writes them and `load` reads them with
  `load_generation_indices`, which raises `MissingGenerationIndicesError`
  for a bundle saved without them.

`test/src/common/datasets/test_generation_indices.py` checks all of this for
every registered dataset class. Tell its `dataset_kwargs` how to build a
small instance of yours; if it reads a CSV, add it to `TABULAR_SOURCES` so
the source-order test covers it too.

If your dataset is synthetic and should vary by dataset realization, make `accepts_seed()` return `True` and add a `seed` argument to `__init__`. The dataset generation script will pass one seed per dataset realization automatically.

### 2. Create an entry in dataset generation config

Create a config file in `extra/conf/scripts/dataset_generation/dataset/` (e.g., `my_dataset.yaml`) to specify how the dataset should be generated:

```yaml
class_name: "MyDataset"
kwargs:
  n_samples: 10000
  # Add other constructor arguments as needed
```

The `class_name` must match the name of your dataset class in the registry, and `kwargs` are passed to the `__init__` method. The file name is the Hydra dataset key used by the pipeline, so the example above is selected with `dataset=my_dataset`.

### 3. Register the dataset class

Add your dataset class to the `REGISTERED_CLASSES` dictionary in `afabench/core/registry.py`:

```python
REGISTERED_CLASSES = {
    # ... existing entries ...
    "MyDataset": "afabench.datasets.datasets.MyDataset",
}
```

This allows the dataset generation script and methods to deserialize your dataset from bundles during training and evaluation via `afabench.core.registry.get_class()`.

### 4. Add to the Snakemake pipeline

List your dataset in one of the dataset configuration files in `extra/workflow/conf/datasets/`. For example, in `extra/workflow/conf/datasets/all.yaml`:

```yaml
datasets:
  - my_dataset
  # ... other datasets ...
```

The pipeline will generate `train.bundle`, `val.bundle`, and `test.bundle` under `extra/output/production/datasets/my_dataset/{dataset_realization_index}/` for each selected dataset realization.

### 5. Add a readable name

(Optional but recommended) Add a display name to `dataset_name_mapping` in `extra/conf/scripts/plotting/common/default.yaml`:

```yaml
dataset_name_mapping:
  # ... existing entries ...
  my_dataset: My Dataset Display Name
```

### 6. Add to dataset sets

(Optional but recommended) Add your dataset to one or more *dataset sets* in `dataset_sets` in `extra/conf/scripts/plotting/common/default.yaml`. Dataset sets group datasets for organized plotting:

```yaml
dataset_sets:
  set1:
    # ... existing datasets ...
    - my_dataset
  all:
    # ... existing datasets ...
    - my_dataset
```

If your dataset is not in any set, the pipeline will still generate and train on it, but plots won't be generated for it. Adding it to the `all` set is typically sufficient.
