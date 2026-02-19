import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Self, cast, final, override
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from ucimlrepo import fetch_ucirepo

from afabench.common.custom_types import AFADataset
from afabench.common.datasets.utils import default_create_subset


def _z_normalize(
    features_df: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    Apply feature-wise Z-normalization using population statistics.

    Zero standard deviations are replaced with 1.0 to avoid division by zero.
    """
    means: pd.Series = cast("pd.Series", features_df.mean())
    stds: pd.Series = cast("pd.Series", features_df.std(ddof=0))
    stds = stds.replace(0, 1.0)
    return (features_df - means) / stds


@final
class CubeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning".

    Implements the AFADataset protocol.
    """

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([20])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([8])

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    def __init__(
        self,
        seed: int = 123,
        n_samples: int = 20000,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std

        # Constants
        self.n_cube_features = 10  # Number of cube features
        self.n_dummy_features = 10  # Remaining features are dummy features

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )
        # Binary codes for labels
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.label_shape[0])
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        x_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=self.rng,
        )

        x_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=self.rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_cube[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        # Concatenate all features
        self.features = torch.cat([x_cube, x_dummy], dim=1)
        assert self.features.shape[1] == self.feature_shape[0]

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

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
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "informative_feature_std": self.informative_feature_std,
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
        obj.seed = data["config"]["seed"]
        obj.n_samples = data["config"]["n_samples"]
        obj.non_informative_feature_mean = data["config"][
            "non_informative_feature_mean"
        ]
        obj.non_informative_feature_std = data["config"][
            "non_informative_feature_std"
        ]
        obj.informative_feature_std = data["config"]["informative_feature_std"]
        obj.n_cube_features = 10
        obj.n_dummy_features = 10
        obj.rng = torch.Generator()
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj


@final
class CubeNonUniformCostsDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A modified version of CUBE that has two blocks of features which both contain informative features, but the blocks have different feature costs.

    This dataset can be used to verify whether an AFA method cares about feature costs.
    """

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([20])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([8])

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    def __init__(
        self,
        seed: int = 123,
        n_samples: int = 20000,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
        cost_scaling: float = 2.0,
    ):
        super().__init__()
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std
        self.cost_scaling = cost_scaling

        # Constants
        self.n_cube_features = 10  # Number of cube features in each block
        self.n_blocks = 2
        # self.n_dummy_features = (
        #     self.feature_shape[0] - self.n_cube_features
        # )  # Remaining features are dummy features

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )
        # Binary codes for labels
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.label_shape[0])
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        x_block1 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=self.rng,
        )

        x_block2 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=self.rng,
        )

        # Insert informative signals, in both blocks
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_block1[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )
            x_block2[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        # Concatenate all features
        self.features = torch.cat([x_block1, x_block2], dim=1)
        assert self.features.shape[1] == self.feature_shape[0]

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

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
                    "seed": self.seed,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "informative_feature_std": self.informative_feature_std,
                    "cost_scaling": self.cost_scaling,
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
        obj.seed = data["config"]["seed"]
        obj.n_samples = data["config"]["n_samples"]
        obj.non_informative_feature_mean = data["config"][
            "non_informative_feature_mean"
        ]
        obj.non_informative_feature_std = data["config"][
            "non_informative_feature_std"
        ]
        obj.informative_feature_std = data["config"]["informative_feature_std"]
        obj.cost_scaling = data["config"]["cost_scaling"]
        obj.n_cube_features = 10
        obj.n_blocks = 2
        obj.rng = torch.Generator()
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj

    @override
    def get_feature_acquisition_costs(self) -> torch.Tensor:
        block1_costs = torch.ones((self.n_cube_features,), dtype=torch.float32)
        block2_costs = self.cost_scaling * torch.ones((self.n_cube_features,))
        combined_costs = torch.cat([block1_costs, block2_costs], dim=-1)

        # Normalize so the average feature cost is still 1.0
        combined_costs = (
            self.n_features * combined_costs / combined_costs.sum()
        )
        return combined_costs

    @property
    def n_features(self) -> int:
        return self.n_cube_features * self.n_blocks


@final
class AFAContextDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A hybrid dataset combining context-based feature selection and the Cube dataset.

    - Features:
        * First n_contexts features: one-hot context (0, 1, ..., n_contexts-1)
        * Next n_contexts * 10 features: Each block of 10 features is informative if context == block index, else noise

    - Label:
        * One of 8 classes encoded by a 3-bit binary vector inserted into the relevant block

    Optimal policy: query the context first, then only the relevant 10-dimensional block.
    """

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    block_size = 10

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size((self.n_features,))

    @property
    def n_features(self) -> int:
        return self.n_contexts + self.n_contexts * self.block_size

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([8])

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
        n_contexts: int = 3,
        *,
        use_cheap_context_features: bool = False,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_contexts = n_contexts
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.use_cheap_context_features = use_cheap_context_features

        # self.n_features = self.n_contexts + self.n_contexts * self.block_size

        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor
        # Sample context (0, 1, ..., n_contexts-1)
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=self.rng
        )
        context_onehot = torch.nn.functional.one_hot(
            context, num_classes=self.n_contexts
        ).float() + torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=self.rng,
        )  # (n_samples, n_contexts)

        # Sample labels 0-7
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            generator=self.rng,
        )

        # Binary codes for labels (8x3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.label_shape[0])
            ],
            dim=0,
        ).flip(-1)  # (8, 3)

        # Create n_contexts blocks of features, each 10D
        blocks = []
        for _block_context in range(self.n_contexts):
            block = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self.non_informative_feature_std,
                size=(self.n_samples, self.block_size),
                generator=self.rng,
            )
            blocks.append(block)
        blocks = torch.stack(blocks, dim=1)  # (n_samples, n_contexts, 10)

        # Insert informative signal into the correct block based on context
        for i in range(self.n_samples):
            ctx = int(context[i].item())
            label = int(y_int[i].item())
            bin_code = binary_codes[label].float()

            # Select 3 indices in the block to hold the binary code (arbitrary positions)
            insert_idx = [(label + j) % self.block_size for j in range(3)]

            noise = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=self.rng,
            )
            blocks[i, ctx, insert_idx] = bin_code + noise

        # Flatten blocks: (n_samples, n_contexts * 10)
        block_features = blocks.view(self.n_samples, -1)

        # Final feature matrix: context (n_contexts) + all block features (n_contexts * 10)
        self.features = torch.cat(
            [context_onehot, block_features], dim=1
        )  # (n_samples, n_contexts + n_contexts * 10)

        # One-hot labels
        self.labels = torch.nn.functional.one_hot(
            y_int, num_classes=self.label_shape[0]
        ).float()

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return self.features.size(0)

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
                    "seed": self.seed,
                    "context_feature_std": self.context_feature_std,
                    "informative_feature_std": self.informative_feature_std,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                    "n_contexts": self.n_contexts,
                    "use_cheap_context_features": self.use_cheap_context_features,
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
        obj.seed = data["config"]["seed"]
        if "n_contexts" in data["config"]:
            obj.n_contexts = data["config"]["n_contexts"]
        else:  # backwards-compatible path
            obj.n_contexts = 3
        obj.context_feature_std = data["config"].get(
            "context_feature_std", 0.1
        )
        obj.informative_feature_std = data["config"]["informative_feature_std"]
        obj.non_informative_feature_mean = data["config"][
            "non_informative_feature_mean"
        ]
        obj.non_informative_feature_std = data["config"][
            "non_informative_feature_std"
        ]
        obj.use_cheap_context_features = data["config"][
            "use_cheap_context_features"
        ]
        obj.rng = torch.Generator()
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj

    @override
    def get_feature_acquisition_costs(self) -> torch.Tensor:
        context_feature_cost_scaling = (
            1 / self.n_contexts if self.use_cheap_context_features else 1
        )
        context_feature_costs = context_feature_cost_scaling * torch.ones(
            self.n_contexts
        )
        remaining_feature_costs = torch.ones(self.n_features - self.n_contexts)
        all_feature_costs = torch.cat(
            [context_feature_costs, remaining_feature_costs], dim=-1
        )
        return all_feature_costs


@final
class MNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """MNIST dataset wrapped to follow the AFADataset protocol."""

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([784])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "extra/data/misc",
    ):
        super().__init__()
        self.train = train
        self.transform = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download = download
        self.root = root

        self.dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
        # Convert images to features (flatten)
        self.features = torch.stack(
            [x[0].flatten() for x in self.dataset]
        ).float()
        assert self.features.shape[1] == self.feature_shape[0]
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

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
                    "train": self.train,
                    "root": self.root,
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
        obj.train = data["config"]["train"]
        obj.root = data["config"]["root"]
        obj.transform = transforms.ToTensor()
        obj.download = False
        obj.dataset = None
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj


@final
class FashionMNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """FashionMNIST dataset wrapped to follow the AFADataset protocol."""

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([784])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "extra/data/misc",
    ):
        super().__init__()
        self.train = train
        self.transform = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download = download
        self.root = root
        self.dataset = datasets.FashionMNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
        # Convert images to flattened feature vectors
        self.features = torch.stack(
            [x[0].flatten() for x in self.dataset]
        ).float()
        assert self.features.shape[1] == self.feature_shape[0]
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

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
                    "train": self.train,
                    "root": self.root,
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
        obj.train = data["config"]["train"]
        obj.root = data["config"]["root"]
        obj.transform = transforms.ToTensor()
        obj.download = False
        obj.dataset = None
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj


@final
class DiabetesDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Diabetes dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements and indicators for diabetes classification.
    The target variable has 3 classes (0, 1, 2) representing different diabetes outcomes.
    """

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([45])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([3])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        root: str = "extra/data/misc/diabetes.csv",
    ):
        super().__init__()
        self.root = root
        # Check if file exists
        if not Path(self.root).exists():
            msg = f"Diabetes dataset not found at {self.root}"
            raise FileNotFoundError(msg)

        # Load the dataset
        df_dataset = pd.read_csv(self.root)

        # Extract features and labels
        # The last column is the target variable (Outcome)
        features_df = df_dataset.iloc[:, :-1]
        labels_df = df_dataset.iloc[:, -1]

        # Fill missing values then normalize
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.feature_shape[0]
        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

        # Store feature names
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset."""
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Tensor, Tensor]:
        """Return all features and labels."""
        return self.features, self.labels

    @override
    def save(self, path: Path) -> None:
        """Save the dataset to a file."""
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {
                    "root": self.root,
                },
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        """Load a dataset from a file."""
        data = torch.load(path / "dataset.pt")
        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.root = data["config"]["root"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        return obj


@final
class MiniBooNEDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    MiniBooNE dataset wrapped to follow the AFADataset protocol.

    This dataset contains particle physics measurements from the MiniBooNE experiment.
    The target variable has 2 classes (signal and background).
    """

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([50])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        root: str = "extra/data/misc/miniboone.csv",
    ):
        super().__init__()
        self.root = root
        if not Path(self.root).exists():
            msg = f"MiniBooNE dataset not found at {self.root}"
            raise FileNotFoundError(msg)

        df_dataset = pd.read_csv(self.root)

        # Assuming the last column is the binary target variable
        features_df = df_dataset.iloc[:, :-1]
        labels_df = df_dataset.iloc[:, -1]

        # Fill missing then normalize
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.feature_shape[0]

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

        self.feature_names = features_df.columns.tolist()

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
                "feature_names": self.feature_names,
                "config": {
                    "root": self.root,
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
        obj.root = data["config"]["root"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        return obj


@final
class PhysionetDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Physionet dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements from ICU patients.
    The target variable has 2 classes (0, 1) representing different outcomes.
    """

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([41])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        root: str = "extra/data/misc/physionet.csv",
    ):
        super().__init__()
        self.root = root
        if not Path(self.root).exists():
            msg = f"Physionet dataset not found at {self.root}"
            raise FileNotFoundError(msg)

        df_dataset = pd.read_csv(self.root)

        features_df = df_dataset.iloc[:, :-1]
        labels_df = df_dataset.iloc[:, -1]

        # Handle missing values by filling with column means and normalize
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)

        # Check for NaNs after tensor conversion
        if torch.isnan(self.features).any():
            msg = "NaNs detected in features after filling missing values."
            raise ValueError(msg)

        # === Standardize features ===
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features_df.values)

        assert self.features.shape[1] == self.feature_shape[0]

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

        self.feature_names = features_df.columns.tolist()

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
                "feature_names": self.feature_names,
                "config": {
                    "root": self.root,
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
        obj.root = data["config"]["root"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        return obj


@final
class BankMarketingDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([getattr(self, "n_features", 16)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, path: str):
        super().__init__()
        self.path = path

        if not Path(self.path).exists():
            self._fetch_and_save()

        df_data = pd.read_csv(self.path, sep=";")
        target_col = "y" if "y" in df_data.columns else "deposit"
        features_df = df_data.drop(columns=[target_col])
        pd.set_option("future.no_silent_downcasting", True)
        target_series = (
            df_data[target_col].replace({"yes": 1, "no": 0}).astype("int64")
        )

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                le = LabelEncoder()
                features_df[col] = le.fit_transform(
                    features_df[col].astype(str)
                )
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.label_shape[0],
        ).float()
        self.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    def _fetch_and_save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        bank_data = fetch_ucirepo(id=222)
        assert bank_data is not None
        assert bank_data.data is not None
        df_data = pd.concat(
            [bank_data.data.features, bank_data.data.targets],
            axis=1,
        )
        df_data.to_csv(self.path, sep=";", index=False)

    @override
    def __getitem__(self, idx: int):
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
                "feature_names": self.feature_names,
                "config": {"path": self.path},
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        obj = cls.__new__(cls)
        obj.path = data["config"]["path"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        obj.n_features = obj.features.shape[1]
        return obj


@final
class CKDDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([getattr(self, "n_features", 24)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, path: str):
        super().__init__()
        self.path = path

        if not Path(self.path).exists():
            self._fetch_and_save()

        df_data = pd.read_csv(self.path)
        features_df = df_data.iloc[:, :-1].copy()
        target_series = df_data.iloc[:, -1]

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                le = LabelEncoder()
                mask = features_df[col].notna()
                if mask.any():
                    features_df.loc[mask, col] = le.fit_transform(
                        features_df.loc[mask, col].astype(str)
                    )
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.label_shape[0],
        ).float()
        self.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    def _fetch_and_save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        ckd_data = fetch_ucirepo(id=336)
        assert ckd_data is not None
        assert ckd_data.data is not None
        features_df = ckd_data.data.features.copy()
        target_df = ckd_data.data.targets.copy()
        target_series = (
            target_df.iloc[:, 0].astype(str).str.strip().str.lower()
        )
        target_series = target_series.map({"ckd": 1, "notckd": 0})
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        df_data.to_csv(self.path, index=False)

    @override
    def __getitem__(self, idx: int):
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
                "feature_names": self.feature_names,
                "config": {"path": self.path},
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        obj = cls.__new__(cls)
        obj.path = data["config"]["path"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        obj.n_features = obj.features.shape[1]
        return obj


@final
class ACTG175Dataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([getattr(self, "n_features", 23)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, path: str):
        super().__init__()
        self.path = path

        if not Path(self.path).exists():
            self._fetch_and_save()

        df_data = pd.read_csv(self.path)
        features_df = df_data.iloc[:, :-1].copy()
        target_series = df_data.iloc[:, -1]

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                le = LabelEncoder()
                mask = features_df[col].notna()
                if mask.any():
                    features_df.loc[mask, col] = le.fit_transform(
                        features_df.loc[mask, col].astype(str)
                    )
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.label_shape[0],
        ).float()
        self.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    def _fetch_and_save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        actg_data = fetch_ucirepo(id=890)
        assert actg_data is not None
        assert actg_data.data is not None
        features_df = actg_data.data.features.copy()
        target_df = actg_data.data.targets.copy()
        target_series = target_df.iloc[:, 0].astype(int)
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        df_data.to_csv(self.path, index=False)

    @override
    def __getitem__(self, idx: int):
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
                "feature_names": self.feature_names,
                "config": {"path": self.path},
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        obj = cls.__new__(cls)
        obj.path = data["config"]["path"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        obj.n_features = obj.features.shape[1]
        return obj


@final
class FICODataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    FICO HELOC credit-risk dataset used in MA-learn experiments.

    Missing values are encoded as {-7, -8, -9} in the raw data.
    """

    # Public mirrors for HELoC challenge data.
    DOWNLOAD_URLS: ClassVar[tuple[str, ...]] = (
        "https://raw.githubusercontent.com/deburky/"
        "boosting-scorecards/main/heloc_dataset_v1.csv",
        "https://github.com/deburky/boosting-scorecards/raw/main/"
        "heloc_dataset_v1.csv",
    )

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([getattr(self, "n_features", 23)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, path: str = "extra/data/misc/fico.csv"):
        super().__init__()
        self.path = path

        if not Path(self.path).exists():
            self._fetch_and_save()

        df_data = pd.read_csv(self.path)
        target_col = "RiskPerformance"
        if target_col not in df_data.columns:
            msg = (
                f"Expected target column '{target_col}' in FICO data at "
                f"{self.path}, got columns={list(df_data.columns)}."
            )
            raise ValueError(msg)

        # In the original challenge data, these values represent missingness.
        df_data = df_data.replace([-7, -8, -9], np.nan)
        features_df = df_data.drop(columns=[target_col]).copy()
        assert df_data is not None, (
            "Failed to load FICO dataset after download."
        )
        assert features_df is not None, (
            "Failed to extract features from FICO dataset."
        )
        target_series = LabelEncoder().fit_transform(
            df_data[target_col].astype(str).str.strip()
        )
        assert target_series is not None, (
            "Failed to extract target from FICO dataset."
        )
        target_series.astype(np.int64)

        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series, dtype=torch.long),
            num_classes=self.label_shape[0],
        ).float()
        self.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    def _fetch_and_save(self) -> None:
        output_path = Path(self.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        errors: list[str] = []
        for url in self.DOWNLOAD_URLS:
            error = self._download_candidate(url, output_path)
            if error is None:
                return
            errors.append(f"{url}: {error}")

        openml_error = self._download_from_openml(output_path)
        if openml_error is None:
            return
        errors.append(f"openml: {openml_error}")

        msg = (
            "Failed to download FICO data automatically. Please download "
            "'heloc_dataset_v1.csv' manually and place it at "
            f"{self.path}. Errors: {' | '.join(errors)}"
        )
        raise FileNotFoundError(msg)

    @staticmethod
    def _is_valid_download(path: Path) -> bool:
        try:
            df_data = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            return False
        return "RiskPerformance" in df_data.columns

    def _download_candidate(self, url: str, output_path: Path) -> str | None:
        try:
            urllib.request.urlretrieve(url, output_path)  # noqa: S310
        except (OSError, URLError, HTTPError, ValueError) as exc:
            return str(exc)

        if not self._is_valid_download(output_path):
            return "Downloaded file is not a valid FICO HELOC dataset."
        return None

    def _download_from_openml(self, output_path: Path) -> str | None:
        openml_ids = (45554, 45026)
        last_error = "unknown error"
        data_home = str(Path(self.path).parent / ".sklearn")
        for data_id in openml_ids:
            try:
                raw_features, raw_target = cast(
                    "tuple[object, object]",
                    fetch_openml(
                        data_id=data_id,
                        as_frame=True,
                        data_home=data_home,
                        parser="auto",
                        return_X_y=True,
                    ),
                )
            except (OSError, URLError, HTTPError, ValueError) as exc:
                last_error = f"data_id={data_id}: {exc}"
                continue

            if isinstance(raw_features, pd.DataFrame):
                features = raw_features
            else:
                features = pd.DataFrame(raw_features)

            if isinstance(raw_target, pd.DataFrame):
                if raw_target.shape[1] != 1:
                    last_error = (
                        f"data_id={data_id}: target has "
                        f"{raw_target.shape[1]} columns"
                    )
                    continue
                target_series = raw_target.iloc[:, 0]
            elif isinstance(raw_target, pd.Series):
                target_series = raw_target
            else:
                target_series = pd.Series(raw_target)
            target_series = target_series.rename("RiskPerformance")

            downloaded = features.copy()
            downloaded["RiskPerformance"] = target_series.to_numpy()
            downloaded.to_csv(output_path, index=False)
            if self._is_valid_download(output_path):
                return None
            last_error = f"data_id={data_id}: downloaded but failed validation"

        return last_error

    @override
    def __getitem__(self, idx: int):
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
                "feature_names": self.feature_names,
                "config": {"path": self.path},
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        obj = cls.__new__(cls)
        obj.path = data["config"]["path"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        obj.n_features = obj.features.shape[1]
        return obj


@final
class PharyngitisDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Pharyngitis RADT dataset used in MA-learn experiments.

    Expected columns follow the naming in the original supplementary dataset.
    """

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "age_y",
        "temperature",
        "swollenadp",
        "pain",
        "tender",
        "tonsillarswelling",
        "exudate",
        "sudden",
        "cough",
        "rhinorrhea",
        "conjunctivitis",
        "headache",
        "erythema",
        "petechiae",
        "abdopain",
        "diarrhea",
        "nauseavomit",
        "scarlet",
    )
    TARGET_COLUMN: ClassVar[str] = "radt"

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([getattr(self, "n_features", 18)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, path: str = "extra/data/misc/pharyngitis.xls") -> None:
        super().__init__()
        self.path = path
        data_path = Path(self.path)
        if not data_path.exists():
            msg = (
                "Pharyngitis dataset file not found. Please download the "
                "supplementary 'minimal dataset' from Miyagi (PLOS ONE) and "
                f"place it at {self.path}."
            )
            raise FileNotFoundError(msg)

        df_data = pd.read_excel(data_path)
        if "number" in df_data.columns:
            df_data = df_data.drop(columns=["number"])

        missing_columns = [
            col
            for col in (*self.FEATURE_COLUMNS, self.TARGET_COLUMN)
            if col not in df_data.columns
        ]
        if missing_columns:
            msg = (
                "Pharyngitis dataset is missing required columns: "
                f"{missing_columns}. Found columns={list(df_data.columns)}."
            )
            raise ValueError(msg)

        features_df = df_data[list(self.FEATURE_COLUMNS)].copy()
        target_series = cast("pd.Series", df_data[self.TARGET_COLUMN])

        for col in features_df.columns:
            column_series = cast("pd.Series", features_df[col])
            if column_series.dtype == "object":
                encoder = LabelEncoder()
                observed = column_series.notna()
                if bool(observed.any()):
                    features_df.loc[observed, col] = encoder.fit_transform(
                        column_series.loc[observed].astype(str)
                    )

        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        target_values = target_series.astype(str).str.strip()
        target_encoded = np.asarray(
            LabelEncoder().fit_transform(target_values),
            dtype=np.int64,
        )

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_encoded, dtype=torch.long),
            num_classes=self.label_shape[0],
        ).float()
        self.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int):
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
                "feature_names": self.feature_names,
                "config": {"path": self.path},
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        obj = cls.__new__(cls)
        obj.path = data["config"]["path"]
        obj.features = data["features"]
        obj.labels = data["labels"]
        obj.feature_names = data["feature_names"]
        obj.n_features = obj.features.shape[1]
        return obj


@final
class ImagenetteDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Imagenette dataset from the FastAI image classification benchmark.

    A subset of 10 easily classified classes from Imagenet.
    """

    IMAGENETTE_URL: ClassVar[str] = (
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([3, 224, 224])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        data_root: str = "extra/data/",
        variant_dir: str = "imagenette2-320",
        load_subdirs: tuple[str, ...] = ("train", "val"),
        image_size: int = 224,
        split_role: str | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.variant_dir = variant_dir
        self.load_subdirs = load_subdirs
        self.image_size = image_size
        self.split_role = split_role  # "train" | "val" | "test"

        use_train_aug = self.split_role == "train"
        self.transform = (
            self._train_transform()
            if use_train_aug
            else self._eval_transform()
        )

        root = self._root()
        sub_datasets: list[ImageFolder] = []
        for sub in self.load_subdirs:
            d = root / sub
            if not d.exists():
                msg = f"Expected subdir '{sub}' at {d}"
                raise FileNotFoundError(msg)
            sub_datasets.append(ImageFolder(str(d), transform=None))

        self.samples = [
            path for ds in sub_datasets for (path, _) in ds.samples
        ]
        self.targets = torch.tensor(
            [y for ds in sub_datasets for (_, y) in ds.samples],
            dtype=torch.long,
        )
        self.indices = torch.arange(len(self.samples), dtype=torch.long)

    def _resolve_index(self, i: int) -> int:
        if self.indices.ndim != 1:
            msg = f"indices must be 1D, got shape {tuple(self.indices.shape)}"
            raise ValueError(msg)
        if i < 0 or i >= int(self.indices.numel()):
            msg = f"Index {i} out of range for dataset of length {len(self)}"
            raise IndexError(msg)
        return int(self.indices[i].item())

    def _train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def _eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def _download_imagenette(self, root: Path) -> None:
        root_parent = root.parent
        root_parent.mkdir(parents=True, exist_ok=True)
        archive_path = root_parent / f"{self.variant_dir}.tgz"

        parsed = urlparse(self.IMAGENETTE_URL)
        if parsed.scheme not in ("http", "https"):
            msg = f"Invalid URL scheme in IMAGENETTE_URL: {parsed.scheme}"
            raise ValueError(msg)

        if not archive_path.exists():
            print(
                f"Downloading Imagenette from {self.IMAGENETTE_URL} to {
                    archive_path
                }"
            )
            urllib.request.urlretrieve(self.IMAGENETTE_URL, archive_path)  # noqa: S310

        print(f"Extracting {archive_path} to {root_parent}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root_parent, filter="data")

    def _root(self) -> Path:
        root = Path(self.data_root) / self.variant_dir
        if (not root.exists()) or (not any(root.iterdir())):
            self._download_imagenette(root)

        if not root.exists():
            msg = (
                f"Imagenette folder not found at {
                    root
                } after attempted download. "
                f"Expected '{self.variant_dir}' with train/ and val/ subdirs."
            )
            raise FileNotFoundError(msg)
        return root

    @override
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        assert (  # noqa: PT018
            self.samples is not None
            and self.targets is not None
            and self.transform is not None
        ), "Dataset not properly initialized"
        idx = self._resolve_index(i)
        path = self.samples[idx]
        y = self.targets[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            x = self.transform(img_rgb)
            # Ensure x is a Tensor (transforms.ToTensor() should guarantee this)
            if not isinstance(x, Tensor):
                msg = f"Transform did not return a Tensor, got {type(x)}"
                raise TypeError(msg)
        # Convert label to one-hot tensor
        y_onehot = torch.nn.functional.one_hot(
            y, num_classes=self.label_shape[0]
        ).float()
        return x, y_onehot

    @override
    def __len__(self) -> int:
        return int(self.indices.numel())

    @override
    def get_all_data(self) -> tuple[Tensor, Tensor]:
        """Return all data as tensors. Note: This loads all images into memory."""
        features = []
        labels = []
        for i in range(len(self)):
            x, y = self[i]
            features.append(x)
            labels.append(y)
        return torch.stack(features), torch.stack(labels)

    @override
    def save(self, path: Path) -> None:
        """Save only the split indices and the dataset config reconstruct later from raw files on load."""
        torch.save(
            {
                "indices": self.indices,
                "config": {
                    "data_root": self.data_root,
                    "variant_dir": self.variant_dir,
                    "load_subdirs": self.load_subdirs,
                    "image_size": self.image_size,
                    "split_role": self.split_role,
                },
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        cfg = data["config"]
        idx = data["indices"]
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, dtype=torch.long)
        idx = idx.to(dtype=torch.long)
        if "split_role" not in cfg or cfg["split_role"] is None:
            msg = "Split role not initialized!"
            raise ValueError(msg)

        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.data_root = cfg["data_root"]
        obj.variant_dir = cfg["variant_dir"]
        obj.load_subdirs = cfg["load_subdirs"]
        obj.image_size = cfg["image_size"]
        obj.split_role = cfg["split_role"]

        use_train_aug = obj.split_role == "train"
        obj.transform = (
            obj._train_transform() if use_train_aug else obj._eval_transform()  # noqa: SLF001
        )

        root = obj._root()  # noqa: SLF001
        sub_datasets: list[ImageFolder] = []
        for sub in obj.load_subdirs:
            d = root / sub
            if not d.exists():
                msg = f"Expected subdir '{sub}' at {d}"
                raise FileNotFoundError(msg)
            sub_datasets.append(ImageFolder(str(d), transform=None))

        all_samples = [path for ds in sub_datasets for (path, _) in ds.samples]
        all_targets = torch.tensor(
            [y for ds in sub_datasets for (_, y) in ds.samples],
            dtype=torch.long,
        )

        # Apply subset filtering
        obj.samples = all_samples
        obj.targets = all_targets
        obj.indices = idx
        n = len(obj.samples)
        if int(obj.indices.numel()) > 0 and (
            int(obj.indices.min().item()) < 0
            or int(obj.indices.max().item()) >= n
        ):
            msg = f"Loaded indices out of bounds: valid [0, {n - 1}]"
            raise ValueError(msg)

        return obj


@final
class SyntheticMNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Synthetic MNIST-like dataset with the same shape as MNIST (28x28 = 784 features, 10 classes).

    Generates synthetic image-like data with patterns that can be learned.
    The left half (14 pixels) contains only noise, while the right half contains
    class-specific patterns for label identification.

    Implements the AFADataset protocol.
    """

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size(
            [1, 28, 28]
        )  # (channels, height, width) - proper image format

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])  # 10 classes, same as MNIST

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        return default_create_subset(self, indices)

    def __init__(  # noqa: C901, PLR0915, PLR0912
        self,
        seed: int = 123,
        # Memory-friendly default (60k samples ≈ 200MB RAM)
        n_samples: int = 10000,
        noise_std: float = 0.1,
        pattern_intensity: float = 0.8,
    ):
        super().__init__()
        self.seed = seed
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.pattern_intensity = pattern_intensity

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # Generate labels uniformly
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )

        # Initialize features with noise in image format (C, H, W)
        self.features = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=(self.n_samples, 1, 28, 28),
            generator=self.rng,
        )

        # Pre-compute coordinate grids for optimized pattern generation
        y_coords, x_coords = torch.meshgrid(
            torch.arange(28, dtype=torch.float32),
            torch.arange(28, dtype=torch.float32),
            indexing="ij",
        )

        # Add class-specific patterns to make the data learnable (right half only)
        for i in range(self.n_samples):
            label = int(y_int[i].item())

            # Create simple geometric patterns for each class
            img = self.features[i, 0]  # Get the single channel (28, 28)

            # Ensure left half (0:14) remains pure noise - regenerate it
            img[:, 0:14] = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=(28, 14),
                generator=self.rng,
            )

            if label == 0:  # Circle-like pattern (right half only)
                center_y, center_x = 14.0, 21.0  # Center in right half
                dist = torch.sqrt(
                    (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
                )
                circle_mask = (dist >= 4) & (
                    dist <= 6
                )  # Smaller circle for half space
                # Only apply to right half
                circle_mask = circle_mask & (x_coords >= 14)
                img[circle_mask] += self.pattern_intensity

            elif label == 1:  # Vertical line (right half)
                img[:, 19:23] += self.pattern_intensity  # Moved to right half

            elif label == 2:  # Horizontal line (right half only)
                img[12:16, 14:] += self.pattern_intensity  # Only right half

            elif label == 3:  # Diagonal line (right half)
                # Create diagonal in right half only
                for j in range(14):
                    if j < 28 and (j + 14) < 28:
                        img[j, j + 14] += self.pattern_intensity

            elif label == 4:  # Square (right half)
                img[8:20, 16:26] = (
                    self.pattern_intensity
                )  # Moved to right half
                img[8:10, 16:26] = self.pattern_intensity
                img[18:20, 16:26] = self.pattern_intensity
                img[8:20, 16:18] = self.pattern_intensity
                img[8:20, 24:26] = self.pattern_intensity

            elif label == 5:  # Cross pattern (right half)
                img[:, 19:23] = (
                    self.pattern_intensity
                )  # Vertical line in right half
                img[12:16, 14:] = (
                    self.pattern_intensity
                )  # Horizontal line in right half

            elif label == 6:  # Triangle-like (right half)
                for y in range(6, 22):
                    width = (y - 6) // 2
                    start_x = max(14, 21 - width)  # Centered in right half
                    end_x = min(28, 21 + width + 1)
                    img[y, start_x:end_x] += self.pattern_intensity

            elif label == 7:  # L-shape (right half)
                img[6:22, 16:20] = (
                    self.pattern_intensity
                )  # Vertical part in right half
                img[18:22, 16:26] = (
                    self.pattern_intensity
                )  # Horizontal part in right half

            elif label == 8:  # X pattern (right half)
                # Main diagonal in right half
                for j in range(14):
                    if j < 28 and (j + 14) < 28:
                        img[j, j + 14] += self.pattern_intensity
                # Anti-diagonal in right half
                for j in range(14):
                    if j < 28 and (27 - j) >= 14:
                        img[j, 27 - j] += self.pattern_intensity

            elif label == 9:  # Dot pattern (right half only)
                img[6:8, 16:18] += (
                    self.pattern_intensity
                )  # Top row, right half
                img[6:8, 20:22] += self.pattern_intensity
                img[6:8, 24:26] += self.pattern_intensity
                img[10:12, 16:18] += (
                    self.pattern_intensity
                )  # Second row, right half
                img[10:12, 20:22] += self.pattern_intensity
                img[10:12, 24:26] += self.pattern_intensity
                img[14:16, 16:18] += (
                    self.pattern_intensity
                )  # Third row, right half
                img[14:16, 20:22] += self.pattern_intensity
                img[14:16, 24:26] += self.pattern_intensity
                img[18:20, 16:18] += (
                    self.pattern_intensity
                )  # Fourth row, right half
                img[18:20, 20:22] += self.pattern_intensity
                img[18:20, 24:26] += self.pattern_intensity

            # Update the channel
            self.features[i, 0] = img

        # Normalize features to [0, 1] range like MNIST
        self.features = torch.clamp(self.features, 0, 1)

        # Convert labels to one-hot
        self.labels = torch.nn.functional.one_hot(
            y_int, num_classes=self.label_shape[0]
        ).float()
        assert self.labels.shape[1] == self.label_shape[0]

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
                    "seed": self.seed,
                    "noise_std": self.noise_std,
                    "pattern_intensity": self.pattern_intensity,
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
        obj.seed = data["config"]["seed"]
        obj.n_samples = data["config"]["n_samples"]
        obj.noise_std = data["config"]["noise_std"]
        obj.pattern_intensity = data["config"]["pattern_intensity"]
        obj.rng = torch.Generator()
        obj.features = data["features"]
        obj.labels = data["labels"]
        return obj
