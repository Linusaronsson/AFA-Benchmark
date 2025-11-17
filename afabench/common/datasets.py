import os
import torch
import zipfile
import urllib.request
import pandas as pd

from PIL import Image
from pathlib import Path
from torch import Tensor
from ucimlrepo import fetch_ucirepo
from torch.utils.data import Dataset
from collections.abc import Callable
from torch.nn import functional as F
from typing import Self, final, override
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder

from afabench.common.custom_types import (
    AFADataset,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


@final
class Shim2018CubeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """The Cube dataset, as described in the paper "Minimizing data consumption with sequential online feature selection" (https://doi.org/10.1007/s13042-012-0092-x)."""

    n_classes = 8
    n_features = 20  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        sigma: float = 0.1,
        seed: int = 123,
    ):
        self.n_samples = n_samples
        self.sigma = sigma
        self.seed = seed
        self.rng = torch.Generator()

    @override
    def generate_data(self):
        self.rng.manual_seed(self.seed)

        # Each coordinate is drawn from a Bernoulli distribution with p=0.5, which is the same as uniform
        coords = torch.randint(
            low=0,
            high=2,
            size=(self.n_samples, 3),
            dtype=torch.int64,
            generator=self.rng,
        )
        # Each corner in the cube is a different label
        labels = torch.einsum(
            "bi,i->b", coords, torch.tensor([1, 2, 4], dtype=torch.int64)
        )
        # Add Gaussian noise to coords
        coords = coords.float()
        coords += (
            torch.randn(
                self.n_samples, 3, dtype=torch.float32, generator=self.rng
            )
            * self.sigma
        )
        # The final features are the coordinates offset according to the labels, and uniform noise for all other features
        self.features = torch.zeros(
            self.n_samples, self.n_features, dtype=torch.float32
        )
        for i in range(self.n_samples):
            offset: int = int(labels[i].item())
            self.features[i, offset : offset + 3] += coords[i]
            # uniform noise on all other features
            self.features[i, :offset] = torch.rand(
                (1, offset), dtype=torch.float32, generator=self.rng
            )
            self.features[i, offset + 3 :] = torch.rand(
                (1, self.n_features - offset - 3),
                dtype=torch.float32,
                generator=self.rng,
            )
        # Convert labels to one-hot encoding
        self.labels = torch.nn.functional.one_hot(
            labels, num_classes=8
        ).float()

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
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
                    "sigma": self.sigma,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning".

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 20  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std

        # Constants
        self.n_cube_features = 10  # Number of cube features
        self.n_dummy_features = (
            self.n_features - self.n_cube_features
        )  # Remaining features are dummy features

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.n_classes,
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        X_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=self.rng,
        )

        X_dummy = torch.normal(
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
            X_cube[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        # Concatenate all features
        self.features = torch.cat([X_cube, X_dummy], dim=1)
        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeSimpleDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A simplified version of the cube dataset, made for debugging purposes. Three features contain label information and three others are noise.

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 6

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.n_classes,
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        self.features = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, 6),
            generator=self.rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            self.features[i, [0, 2, 4]] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
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
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class CubeOnlyInformativeDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    A version of the cube dataset that only has the first 10 informative features.

    Implements the AFADataset protocol.
    """

    n_classes = 8
    n_features = 10  # Fixed number of features

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        informative_feature_std: float = 0.1,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.informative_feature_std = informative_feature_std

        self.rng = torch.Generator()

    @override
    def generate_data(self) -> None:
        self.rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.n_classes,
            (self.n_samples,),
            dtype=torch.int64,
            generator=self.rng,
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)

        # Initialize feature blocks
        self.features = torch.zeros(
            size=(self.n_samples, self.n_features),
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = binary_codes[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            self.features[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=self.rng,
                )
                + mu_bin
            )

        assert self.features.shape[1] == self.n_features

        # Labels
        self.labels = y_int
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[MaskedFeatures, FeatureMask]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[MaskedFeatures, FeatureMask]:
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
                    "informative_feature_std": self.informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


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

    n_classes = 8
    block_size = 10
    n_features = 33

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_contexts = 3
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std

        # self.n_features = self.n_contexts + self.n_contexts * self.block_size

        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor

    @override
    def generate_data(self) -> None:
        # Sample context (0, 1, ..., n_contexts-1)
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=self.rng
        )
        context_onehot = F.one_hot(
            context, num_classes=self.n_contexts
        ).float() + torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=self.rng,
        )  # (n_samples, n_contexts)

        # Sample labels 0–7
        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), generator=self.rng
        )

        # Binary codes for labels (8×3)
        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)  # (8, 3)

        # Create n_contexts blocks of features, each 10D
        blocks = []
        for block_context in range(self.n_contexts):
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
        self.labels = F.one_hot(y_int, num_classes=self.n_classes).float()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
                    # "n_contexts": self.n_contexts,
                    "informative_feature_std": self.informative_feature_std,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class AFAContextRandomInsertDataset(
    Dataset[tuple[Tensor, Tensor]], AFADataset
):
    """
    A hybrid dataset with context-based feature selection and Cube-like structure.

    - Features:
        * First n_contexts features: one-hot context
        * Next n_contexts * 10 features: only the block corresponding to the context is informative

    - Labels:
        * One of 8 classes encoded by a 3-bit binary vector inserted into the relevant block
          at **random** positions for each sample.

    Optimal policy: query the context, then the corresponding block.
    """

    n_classes = 8
    block_size = 10
    n_features = 33

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_contexts = 3
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std

        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor

    @override
    def generate_data(self) -> None:
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=self.rng
        )
        context_onehot = F.one_hot(
            context, num_classes=self.n_contexts
        ).float()
        context_onehot += torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=self.rng,
        )

        y_int = torch.randint(
            0, self.n_classes, (self.n_samples,), generator=self.rng
        )

        binary_codes = torch.stack(
            [
                torch.tensor([int(b) for b in format(i, "03b")])
                for i in range(self.n_classes)
            ],
            dim=0,
        ).flip(-1)  # (8, 3)

        blocks = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_contexts, self.block_size),
            generator=self.rng,
        )

        for i in range(self.n_samples):
            ctx = int(context[i])
            label = int(y_int[i])
            bin_code = binary_codes[label].float()

            # Select 3 **random** indices for informative positions
            insert_idx = torch.randperm(self.block_size, generator=self.rng)[
                :3
            ]

            noise = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=self.rng,
            )
            blocks[i, ctx, insert_idx] = bin_code + noise

        block_features = blocks.view(self.n_samples, -1)

        self.features = torch.cat([context_onehot, block_features], dim=1)
        self.labels = F.one_hot(y_int, num_classes=self.n_classes).float()

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
                    "informative_feature_std": self.informative_feature_std,
                    "non_informative_feature_mean": self.non_informative_feature_mean,
                    "non_informative_feature_std": self.non_informative_feature_std,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class ContextSelectiveXORDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Each data point:
      - context c ∈ {0, 1}
      - features: x1, x2, x3, x4 ∈ {0, 1}

    Label:
      - if c == 0: label = x1 XOR x2
      - if c == 1: label = x3 XOR x4

    Optimal strategy: query context first, then only relevant pair.
    """

    n_classes = 2
    n_features = 5

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self.features: Tensor
        self.labels: Tensor

        self.generate_data()

    @override
    def generate_data(self) -> None:
        # Sample context bit: 0 or 1
        context = torch.randint(0, 2, (self.n_samples,), generator=self.rng)

        # Sample binary features: x1, x2, x3, x4
        x = torch.randint(0, 2, (self.n_samples, 4), generator=self.rng)

        labels = torch.zeros(self.n_samples, dtype=torch.int64)
        for i in range(self.n_samples):
            if context[i] == 0:
                labels[i] = int(x[i, 0] ^ x[i, 1])  # x1 XOR x2
            else:
                labels[i] = int(x[i, 2] ^ x[i, 3])  # x3 XOR x4

        # One-hot labels
        self.labels = torch.nn.functional.one_hot(
            labels, num_classes=2
        ).float()

        # Combine context and features into full feature vector
        self.features = torch.cat(
            [
                context.unsqueeze(1).float(),  # shape: (n_samples, 1)
                x.float(),  # shape: (n_samples, 4)
            ],
            dim=1,
        )  # shape: (n_samples, 5)

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self):
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
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        cfg = data["config"]
        ds = cls(**cfg)
        ds.features = data["features"]
        ds.labels = data["labels"]
        return ds


@final
class MNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """MNIST dataset wrapped to follow the AFADataset protocol."""

    n_classes = 10
    n_features = 784  # Fixed number of features (28x28 images flattened)

    def __init__(
        self,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "extra/data",
        seed: None = None,  # does nothing, added to not panic during data generation
    ):
        super().__init__()
        self.train = train
        self.transform = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download = download
        self.root = root

        self.dataset = None  # set when generating data

    @override
    def generate_data(self) -> None:
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
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class FashionMNISTDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """FashionMNIST dataset wrapped to follow the AFADataset protocol."""

    n_classes = 10
    n_features = 784  # 28x28 flattened images

    def __init__(
        self,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "extra/data",
        seed: None = None,  # does nothing, added to not panic during data generation
    ):
        super().__init__()
        self.train = train
        self.transform = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download = download
        self.root = root
        self.dataset = None  # set when generating data

    @override
    def generate_data(self) -> None:
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
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor([x[1] for x in self.dataset])
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return self.features.size(0)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        return dataset


@final
class DiabetesDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Diabetes dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements and indicators for diabetes classification.
    The target variable has 3 classes (0, 1, 2) representing different diabetes outcomes.
    """

    n_classes = 3
    n_features = 45

    def __init__(
        self,
        root: str = "extra/data/misc/diabetes.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None  # set when generating data
        self.feature_costs = None

    @override
    def generate_data(self) -> None:
        """Load and preprocess the diabetes dataset."""
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Check if file exists
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"Diabetes dataset not found at {self.root}"
            )

        # Load the dataset
        df = pd.read_csv(self.root)

        # Extract features and labels
        # The last column is the target variable (Outcome)
        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.n_features
        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        # Store feature names
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single sample from the dataset."""
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
                "feature_costs": self.feature_costs,
                "config": {
                    "root": self.root,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        """Load a dataset from a file."""
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        dataset.feature_costs = data["feature_costs"]
        return dataset

    @override
    def get_feature_acquisition_costs(self) -> Tensor:
        # msg = "Missing feature acquisition costs for DiabetesDataset."
        # raise NotImplementedError(msg)
        if self.feature_costs is None:
            raise ValueError(
                "Missing feature acquisition costs for DiabetesDataset. Generate or load costs first."
            )
        else:
            return self.feature_costs


@final
class MiniBooNEDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    MiniBooNE dataset wrapped to follow the AFADataset protocol.

    This dataset contains particle physics measurements from the MiniBooNE experiment.
    The target variable has 2 classes (signal and background).
    """

    n_classes = 2
    n_features = 50

    def __init__(
        self,
        root: str = "extra/data/misc/miniboone.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None  # set when generating data

    @override
    def generate_data(self) -> None:
        """Load and preprocess the MiniBooNE dataset."""
        torch.manual_seed(self.seed)

        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"MiniBooNE dataset not found at {self.root}"
            )

        df = pd.read_csv(self.root)

        # Assuming the last column is the binary target variable
        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        assert self.features.shape[1] == self.n_features

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class PhysionetDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Physionet dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements from ICU patients.
    The target variable has 2 classes (0, 1) representing different outcomes.
    """

    n_classes = 2
    n_features = 41

    def __init__(
        self,
        root: str = "extra/data/misc/physionet.csv",
        seed: int = 123,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None  # set when generating data

    @override
    def generate_data(self) -> None:
        """Load and preprocess the Physionet dataset."""
        torch.manual_seed(self.seed)

        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"Physionet dataset not found at {self.root}"
            )

        df = pd.read_csv(self.root)

        features_df = df.iloc[:, :-1]
        labels_df = df.iloc[:, -1]

        # Handle missing values by filling with column means
        features_df = features_df.fillna(features_df.mean())

        # Convert to tensors
        # self.features = torch.tensor(scaled_features, dtype=torch.float32)
        self.features = torch.tensor(features_df.values, dtype=torch.float32)

        # Check for NaNs after tensor conversion
        if torch.isnan(self.features).any():
            raise ValueError(
                "NaNs detected in features after filling missing values."
            )

        # === Standardize features ===
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features_df.values)

        assert self.features.shape[1] == self.n_features

        self.labels = torch.tensor(labels_df.values, dtype=torch.long)
        self.labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.n_classes
        ).float()
        assert self.labels.shape[1] == self.n_classes

        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
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
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class BankMarketingDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    n_classes = 2
    n_features = 16

    def __init__(
        self,
        root: str = "extra/data/misc/bank_marketing.csv",
        seed: int = 123,
        # For catching API consistency
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None

    @override
    def generate_data(self) -> None:
        torch.manual_seed(self.seed)

        if not os.path.exists(self.root):
            self._download_dataset()

        df = pd.read_csv(self.root, sep=";")

        # Process data
        target_col = "y" if "y" in df.columns else "deposit"
        features_df = df.drop(columns=[target_col])
        target_series = df[target_col].map({"yes": 1, "no": 0})

        # Encode categoricals
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                le = LabelEncoder()
                features_df[col] = le.fit_transform(
                    features_df[col].astype(str)
                )

        features_df = features_df.fillna(features_df.mean())

        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.n_classes,
        ).float()

        BankMarketingDataset.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

        # Clean up raw CSV
        os.remove(self.root)

    def _download_dataset(self):
        os.makedirs(os.path.dirname(self.root), exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        zip_path = os.path.join(os.path.dirname(self.root), "bank.zip")

        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_file = next(
                f for f in zip_ref.namelist() if f.endswith("bank-full.csv")
            )
            zip_ref.extract(csv_file, os.path.dirname(self.root))
            os.rename(
                os.path.join(os.path.dirname(self.root), csv_file),
                self.root,
            )
        os.remove(zip_path)

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self):
        return self.features, self.labels

    @override
    def save(self, path: Path):
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {"root": self.root, "seed": self.seed},
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class CKDDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Chronic Kidney Disease dataset from UCI Machine Learning Repository.

    Binary classification task to predict CKD (chronic kidney disease) vs non-CKD.
    Dataset contains 400 patient records with 24 clinical features.
    """

    n_classes = 2
    n_features = 24  # Will be updated after preprocessing

    def __init__(
        self,
        root: str = "extra/data/ckd/chronic_kidney_disease.csv",
        seed: int = 123,
        # Not used, kept for API consistency
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None

    @override
    def generate_data(self) -> None:
        torch.manual_seed(self.seed)

        # Fetch dataset (ID 336 is Chronic Kidney Disease)
        ckd_data = fetch_ucirepo(id=336)

        # Get features and targets as pandas DataFrames
        # Make copies to avoid SettingWithCopyWarning
        features_df = ckd_data.data.features.copy()
        target_df = ckd_data.data.targets.copy()

        # Process target - convert 'ckd'/'notckd' to 1/0
        # Handle potential whitespace and different capitalizations
        target_series = (
            target_df.iloc[:, 0].astype(str).str.strip().str.lower()
        )
        target_series = target_series.map({"ckd": 1, "notckd": 0})

        # Handle any NaN values in target (shouldn't happen, but just in case)
        if target_series.isna().any():
            unique_vals = target_df.iloc[:, 0].unique()
            raise ValueError(
                f"Found NaN values in target after mapping. "
                f"Unique values in original target: {unique_vals}"
            )

        # Handle missing values and encode categorical columns
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                # Use label encoding for categorical variables
                le = LabelEncoder()
                # Handle NaN values before label encoding
                mask = features_df[col].notna()
                if mask.any():
                    features_df.loc[mask, col] = le.fit_transform(
                        features_df.loc[mask, col].astype(str)
                    )

        # Convert all columns to numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        # Fill missing values with mean
        features_df = features_df.fillna(features_df.mean())

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.n_classes,
        ).float()

        # Update n_features based on actual data
        CKDDataset.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self):
        return self.features, self.labels

    @override
    def save(self, path: Path):
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {"root": self.root, "seed": self.seed},
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


@final
class ACTG175Dataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    AIDS Clinical Trials Group Study 175 dataset from UCI Machine Learning Repository.

    Binary classification task to predict death within a certain time window.
    Dataset contains 2,139 HIV-infected patients with 23 clinical features.

    From the famous ACTG 175 clinical trial comparing different HIV treatments.
    """

    n_classes = 2
    n_features = 23  # Will be updated after preprocessing

    def __init__(
        self,
        root: str = "extra/data/actg175/actg175.csv",
        seed: int = 123,
        # Not used, kept for API consistency
        *args,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.feature_names = None

    @override
    def generate_data(self) -> None:
        torch.manual_seed(self.seed)

        # Fetch dataset (ID 890 is AIDS Clinical Trials Group Study 175)
        actg_data = fetch_ucirepo(id=890)

        # Get features and targets as pandas DataFrames
        # Make copies to avoid SettingWithCopyWarning
        features_df = actg_data.data.features.copy()
        target_df = actg_data.data.targets.copy()

        # Process target - binary classification for death
        # The target variable in ACTG175 is typically 'cid' (censor indicator for death)
        # or we may need to check what column name is used
        # Get the first (and likely only) target column
        target_col = target_df.columns[0]
        target_series = target_df[target_col]

        # Convert to binary (0/1) if not already
        if target_series.dtype == "object":
            # Map string values to binary
            unique_vals = target_series.unique()
            if len(unique_vals) == 2:
                # Create mapping for binary classification
                target_series = target_series.map(
                    {unique_vals[0]: 0, unique_vals[1]: 1}
                )
        else:
            # Ensure it's 0/1
            target_series = target_series.astype(int)

        # Handle any NaN values in target
        if target_series.isna().any():
            raise ValueError(
                f"Found NaN values in target. "
                f"Target column: {target_col}, "
                f"Unique values: {target_df[target_col].unique()}"
            )

        # Handle missing values and encode categorical columns
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                # Use label encoding for categorical variables
                le = LabelEncoder()
                # Handle NaN values before label encoding
                mask = features_df[col].notna()
                if mask.any():
                    features_df.loc[mask, col] = le.fit_transform(
                        features_df.loc[mask, col].astype(str)
                    )

        # Convert all columns to numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        # Fill missing values with mean
        features_df = features_df.fillna(features_df.mean())

        # Convert to tensors
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(target_series.values, dtype=torch.long),
            num_classes=self.n_classes,
        ).float()

        # Update n_features based on actual data
        ACTG175Dataset.n_features = self.features.shape[1]
        self.feature_names = features_df.columns.tolist()

    @override
    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self):
        return len(self.features)

    @override
    def get_all_data(self):
        return self.features, self.labels

    @override
    def save(self, path: Path):
        torch.save(
            {
                "features": self.features,
                "labels": self.labels,
                "feature_names": self.feature_names,
                "config": {"root": self.root, "seed": self.seed},
            },
            path,
        )

    @classmethod
    @override
    def load(cls, path: Path):
        data = torch.load(path)
        dataset = cls(**data["config"])
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.feature_names = data["feature_names"]
        return dataset


# @final
class ImagenetteDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Imagenette dataset from the FastAI image classification benchmark.

    A subset of 10 easily classified classes from Imagenet.
    """

    n_classes = 10
    n_features = 3 * 224 * 224

    def __init__(
        self,
        data_root: str = "extra/data/",
        variant_dir: str = "imagenette2-320",
        load_subdirs: tuple[str, ...] = ("train", "val"),
        seed: int = 123,
        image_size: int = 224,
        split_role: str | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.variant_dir = variant_dir
        self.load_subdirs = load_subdirs
        self.seed = seed
        self.image_size = image_size
        self.split_role = split_role  # "train" | "val" | "test"
        self.indices: torch.Tensor | None = None
        self.transform = None
        # self.samples: list[str] | None = None
        # self.targets: torch.Tensor | None = None

        # self.features: Tensor | None = None
        # self.labels: Tensor | None = None
        # self.indices: Tensor | None = None
        # self.class_to_idx: dict[str, int] | None = None
        # self.classes: list[str] | None = None

    def _train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def _eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def _root(self) -> Path:
        root = Path(self.data_root) / self.variant_dir
        if not root.exists():
            raise FileNotFoundError(
                f"Imagenette folder not found at {root}. "
                f"Expected '{self.variant_dir}' with train/ and val/ subdirs."
            )
        return root

    def generate_data(self) -> None:
        torch.manual_seed(self.seed)
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
                raise FileNotFoundError(f"Expected subdir '{sub}' at {d}")
            sub_datasets.append(ImageFolder(str(d), transform=None))

        self.samples = [
            path for ds in sub_datasets for (path, _) in ds.samples
        ]
        self.targets = torch.tensor(
            [y for ds in sub_datasets for (_, y) in ds.samples],
            dtype=torch.long,
        )
        self.indices = torch.arange(len(self.samples), dtype=torch.long)

    def __getitem__(self, idx: int):  # type: ignore[override]
        assert (
            self.samples is not None
            and self.targets is not None
            and self.transform is not None
        )
        path = self.samples[idx]
        y = self.targets[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, y
        # return self.features[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.samples)
        # return self.features.shape[0]

    # TODO need to modify the evaluation script as well
    # def get_all_data(self) -> tuple[Tensor, Tensor]:
    #     return self.features, self.labels

    def save(self, path: Path) -> None:
        """
        Save only the split indices and the dataset config
        reconstruct later from raw files on load
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "indices": self.indices,
                "config": {
                    "data_root": self.data_root,
                    "variant_dir": self.variant_dir,
                    "load_subdirs": self.load_subdirs,
                    "seed": self.seed,
                    "image_size": self.image_size,
                    "split_role": self.split_role,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path)
        cfg = data["config"]
        idx = data["indices"]
        if "split_role" not in cfg or cfg["split_role"] is None:
            stem = Path(path).stem.lower()
            if "train" in stem:
                cfg["split_role"] = "train"
            elif "val" in stem:
                cfg["split_role"] = "val"
            elif "test" in stem:
                cfg["split_role"] = "test"
            else:
                cfg["split_role"] = "val"

        ds = cls(**cfg)
        ds.generate_data()
        # ds.features = ds.features[idx]
        # ds.labels = ds.labels[idx]
        ds.samples = [ds.samples[i] for i in idx.tolist()]
        ds.targets = ds.targets[idx]
        ds.indices = idx
        return ds
