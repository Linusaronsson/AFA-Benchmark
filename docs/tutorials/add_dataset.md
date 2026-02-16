# Adding a New Dataset

## Overview

Datasets in AFA-Benchmark are serialized as **bundles** (directories containing the dataset data and metadata) and are deserialized by methods during training and evaluation. To add a new dataset to the benchmark, you need to define a dataset class, configure how it's generated, register it for deserialization, and integrate it with the pipeline and plotting system.

## Step-by-Step Guide

### 1. Define a dataset class

Define a dataset class in `afabench/common/datasets/datasets.py` that implements the `AFADataset` protocol.

**Minimal example:**


```python
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

    def __init__(
        self,
        n_samples: int
    ):
        super().__init__()
        self.n_samples = n_samples

        self.features = torch.randn(n_samples, 5)
        self.labels = F.one_hot(
            torch.randint(low=0, high=3, size=(self.n_samples,)),
            num_classes=3
        ).float()

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
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
                "config": {
                    "n_samples": self.n_samples,
                }
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
        return obj
```

### 2. Create an entry in dataset generation config

Create a config file in `extra/conf/scripts/dataset_generation/dataset/` (e.g., `my_dataset.yaml`) to specify how the dataset should be generated:

```yaml
class_name: "MyDataset"
kwargs:
  n_samples: 10000
  # Add other constructor arguments as needed
```

The `class_name` must match the name of your dataset class, and `kwargs` are passed to the `__init__` method.

### 3. Register the dataset class

Add your dataset class to the `REGISTERED_CLASSES` dictionary in `afabench/common/registry.py`:

```python
REGISTERED_CLASSES = {
    # ... existing entries ...
    "MyDataset": "afabench.common.datasets.datasets.MyDataset",
}
```

This allows methods to deserialize your dataset from bundles during training and evaluation.

### 4. Add to the Snakemake pipeline

List your dataset in one of the dataset configuration files in `extra/workflow/conf/`. Common options are:

- `datasets_main.yaml` - Main production datasets
- `datasets_full.yaml` - Full set including experimental datasets

For example, in `datasets_main.yaml`:

```yaml
datasets:
  - my_dataset
  # ... other datasets ...
```

### 5. Add a readable name

(Optional but recommended) Add a display name in `DATASET_NAME_MAPPING` in `afabench/eval/plotting_config.py`:

```python
DATASET_NAME_MAPPING = {
    # ... existing entries ...
    "my_dataset": "My Dataset Display Name",
}
```

### 6. Add to dataset sets

(Optional but recommended) Add your dataset to one or more *dataset sets* in `DATASET_SETS` in `afabench/eval/plotting_config.py`. Dataset sets group datasets for organized plotting:

```python
DATASET_SETS = {
    "set1": {
        # ... existing datasets ...
        "my_dataset",
    },
    "all": {
        # ... existing datasets ...
        "my_dataset",
    },
}
```

If your dataset is not in any set, the pipeline will still generate and train on it, but plots won't be generated for it. Adding it to the `"all"` set is typically sufficient.

## See Also

- [Bundle Format](../bundle_format.md) - Details about dataset serialization
- [Pipeline Explanation](./pipeline_explanation.md) - How the full pipeline works
- [Terminology](../terminology.md) - Common terms and concepts
