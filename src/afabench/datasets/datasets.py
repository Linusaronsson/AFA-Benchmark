import copy
import tarfile
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self, cast, final, override
from urllib.parse import urlparse

import pandas as pd
import torch
from jaxtyping import Integer
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, VisionDataset
from ucimlrepo import fetch_ucirepo

from afabench.core.types import AFADataset, Features, Label
from afabench.datasets.feature_costs import load_saved_feature_costs
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import load_generation_indices

type CubeBinaryCodes = Integer[Tensor, "8 3"]
type ClassIndices = Integer[Tensor, "n_instances"]  # noqa: F821

# The 3-bit binary code of each of the 8 CUBE labels, least significant
# bit first
CUBE_BINARY_CODES: CubeBinaryCodes = torch.stack(
    [torch.tensor([int(b) for b in format(i, "03b")]) for i in range(8)],
    dim=0,
).flip(-1)


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


def _one_hot(labels: ClassIndices, n_classes: int) -> Label:
    return torch.nn.functional.one_hot(labels, num_classes=n_classes).float()


@final
class CubeDataset(InMemoryAFADataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning".

    Implements the AFADataset protocol.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "informative_feature_std",
    )
    n_cube_features: ClassVar[int] = 10
    # Remaining features are dummy features
    n_dummy_features: ClassVar[int] = 10

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

    def __init__(
        self,
        seed: int = 123,
        n_samples: int = 20000,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std
        super().__init__()

    @override
    def _build(self) -> tuple[Features, Label]:
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=rng,
        )

        # Initialize feature blocks
        x_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        x_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = CUBE_BINARY_CODES[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_cube[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )

        features = torch.cat([x_cube, x_dummy], dim=1)
        return features, _one_hot(y_int, self.label_shape[0])


@final
class CubeNonUniformCostsDataset(InMemoryAFADataset):
    """
    A modified version of CUBE that has two blocks of features which both contain informative features, but the blocks have different feature costs.

    This dataset can be used to verify whether an AFA method cares about feature costs.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "informative_feature_std",
        "cost_scaling",
    )
    # Number of cube features in each block
    n_cube_features: ClassVar[int] = 10
    n_blocks: ClassVar[int] = 2

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

    def __init__(
        self,
        seed: int = 123,
        n_samples: int = 20000,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
        cost_scaling: float = 2.0,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std
        self.cost_scaling = cost_scaling
        super().__init__()
        self.feature_costs = self._compute_feature_costs()

    @override
    def _build(self) -> tuple[Features, Label]:
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=rng,
        )

        # Initialize feature blocks
        x_block1 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        x_block2 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        # Insert informative signals, in both blocks
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = CUBE_BINARY_CODES[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_block1[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )
            x_block2[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )

        features = torch.cat([x_block1, x_block2], dim=1)
        return features, _one_hot(y_int, self.label_shape[0])

    def _compute_feature_costs(self) -> torch.Tensor:
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
class CubeNMDataset(InMemoryAFADataset):
    """
    Synthetic tabular benchmark implementing the paper's CUBE-NM setup.

    Each sample has a logical context variable that selects which one of
    `n_contexts` CUBE-style 10-feature blocks is informative. The logical
    context is encoded as a noisy one-hot vector in the first `n_contexts`
    feature positions, matching the paper's implementation detail that the
    whole context can be acquired with one grouped action.

    Feature layout:
    - context encoding: `n_contexts` noisy one-hot features
    - block features: `n_contexts` blocks of size `block_size`
      Only the block chosen by the context contains class signal.

    Labels are 8-way one-hot vectors derived from the standard CUBE binary
    code pattern. The canonical benchmark setting uses `n_contexts=5`, which
    gives 55 observed features and 51 acquisition groups under
    `CubeNMUnmasker`.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "context_feature_std",
        "informative_feature_std",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "n_contexts",
        "use_cheap_context_features",
    )
    block_size: ClassVar[int] = 10

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

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

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
        n_contexts: int = 5,
        *,
        use_cheap_context_features: bool = True,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_contexts = n_contexts
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.use_cheap_context_features = use_cheap_context_features
        super().__init__()
        self.feature_costs = self._compute_feature_costs()

    @override
    def _build(self) -> tuple[Features, Label]:
        rng = torch.Generator().manual_seed(self.seed)
        # Sample the logical context, then encode it as a noisy one-hot
        # vector to match the benchmark setup used in the paper.
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=rng
        )
        context_onehot = torch.nn.functional.one_hot(
            context, num_classes=self.n_contexts
        ).float() + torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=rng,
        )  # (n_samples, n_contexts)

        # Sample labels 0-7
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            generator=rng,
        )

        # Create one candidate CUBE block per context value.
        blocks = []
        for _block_context in range(self.n_contexts):
            block = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self.non_informative_feature_std,
                size=(self.n_samples, self.block_size),
                generator=rng,
            )
            blocks.append(block)
        blocks = torch.stack(blocks, dim=1)  # (n_samples, n_contexts, 10)

        # Only the context-selected block carries label information.
        for i in range(self.n_samples):
            ctx = int(context[i].item())
            label = int(y_int[i].item())
            bin_code = CUBE_BINARY_CODES[label].float()

            # Select 3 indices in the block to hold the binary code (arbitrary positions)
            insert_idx = [(label + j) % self.block_size for j in range(3)]

            noise = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=rng,
            )
            blocks[i, ctx, insert_idx] = bin_code + noise

        # Flatten blocks: (n_samples, n_contexts * block_size)
        block_features = blocks.view(self.n_samples, -1)

        # Final matrix: one-hot context features followed by all blocks.
        features = torch.cat(
            [context_onehot, block_features], dim=1
        )  # (n_samples, n_contexts + n_contexts * block_size)
        return features, _one_hot(y_int, self.label_shape[0])

    def _compute_feature_costs(self) -> torch.Tensor:
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


class TorchvisionImageDataset(InMemoryAFADataset, ABC):
    """A torchvision image classification dataset, with every 28x28 image flattened to 784 features and 10 classes."""

    config_keys: ClassVar[tuple[str, ...]] = ("train", "root")

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
        root: str,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
    ):
        self.train: bool = train
        self.root: str = root
        # Used only to build the data, so not part of the config
        self.transform: Callable[[Tensor], Tensor] = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download: bool = download
        super().__init__()

    @abstractmethod
    def _images(self) -> VisionDataset:
        """Return the torchvision dataset of `root`, `train`, `transform` and `download`."""

    @override
    def _build(self) -> tuple[Features, Label]:
        images = self._images()
        features = torch.stack([x[0].flatten() for x in images]).float()
        labels = torch.tensor([x[1] for x in images])
        return features, _one_hot(labels, self.label_shape[0])


@final
class MNISTDataset(TorchvisionImageDataset):
    """MNIST dataset wrapped to follow the AFADataset protocol."""

    def __init__(
        self,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "data/mnist",
    ):
        super().__init__(
            root, train=train, transform=transform, download=download
        )

    @override
    def _images(self) -> VisionDataset:
        return datasets.MNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )


@final
class FashionMNISTDataset(TorchvisionImageDataset):
    """FashionMNIST dataset wrapped to follow the AFADataset protocol."""

    def __init__(
        self,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "data/fashion_mnist",
    ):
        super().__init__(
            root, train=train, transform=transform, download=download
        )

    @override
    def _images(self) -> VisionDataset:
        return datasets.FashionMNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )


class TabularCSVDataset(InMemoryAFADataset, ABC):
    """
    A real-world tabular dataset read from one CSV file, in the file's row order (ADR 0004).

    By default the last column holds the class label and the other columns
    the features; `_read_source` is where a subclass reads its file
    differently, and `_fetch_source` where it downloads a missing file.
    Categorical feature columns are label-encoded, missing values are filled
    with the column mean, and every feature is z-normalized over the rows
    kept. The feature names and each instance's source row are saved in
    dataset.pt beside the tensors.
    """

    feature_names: list[str]
    # Row of each instance in the source file, counting data rows from 0;
    # differs from the generation index once `_read_source` drops rows
    source_rows: Tensor

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, source: Path):
        # Used only to build the data; the subclass's constructor argument
        # it comes from is the config
        self.source: Path = source
        super().__init__()

    def _fetch_source(self) -> None:
        """Write the source file where `source` points, when it is missing."""
        msg = f"{type(self).__name__} source not found at {self.source}"
        raise FileNotFoundError(msg)

    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return the feature columns and the integer class label of each row of `source`.

        A subclass may drop rows but must keep their order and the frames'
        index, which is the source row of each instance.
        """
        frame = pd.read_csv(self.source)
        return frame.iloc[:, :-1].copy(), frame.iloc[:, -1]

    @override
    def _build(self) -> tuple[Features, Label]:
        if not self.source.exists():
            self._fetch_source()
        features_df, labels = self._read_source()

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                mask = features_df[col].notna()
                if bool(mask.any()):
                    features_df.loc[mask, col] = LabelEncoder().fit_transform(
                        features_df.loc[mask, col].astype(str)
                    )
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
        features_df = features_df.fillna(features_df.mean())
        features_df = _z_normalize(features_df)

        features = torch.tensor(features_df.values, dtype=torch.float32)
        if torch.isnan(features).any():
            msg = (
                f"{type(self).__name__} has NaN features after filling "
                f"missing values from {self.source}; a column has no values"
            )
            raise ValueError(msg)
        self.feature_names = features_df.columns.tolist()
        self.source_rows = torch.tensor(
            features_df.index.to_numpy(), dtype=torch.long
        )
        return features, _one_hot(
            torch.tensor(labels.values, dtype=torch.long), self.label_shape[0]
        )

    def get_source_rows(self) -> Tensor:
        """Return the source file row of each instance, in split order."""
        return self.source_rows

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        subset = super().create_subset(indices)
        subset.source_rows = self.source_rows[list(indices)]
        return subset

    @override
    def _saved_data(self) -> dict[str, object]:
        return super()._saved_data() | {
            "feature_names": self.feature_names,
            "source_rows": self.source_rows,
        }

    @override
    def _restore(self, data: Mapping[str, object], file: Path) -> None:
        super()._restore(data, file)
        feature_names = data.get("feature_names")
        if not isinstance(feature_names, list):
            msg = (
                f"Dataset saved at {file} has no feature names, "
                f"got {feature_names!r}"
            )
            raise TypeError(msg)
        self.feature_names = feature_names
        source_rows = data.get("source_rows")
        if not isinstance(source_rows, Tensor) or len(source_rows) != len(
            self.features
        ):
            msg = (
                f"Dataset saved at {file} has no source row per instance, "
                f"got {source_rows!r}"
            )
            raise TypeError(msg)
        self.source_rows = source_rows


@final
class DiabetesDataset(TabularCSVDataset):
    """
    Diabetes dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements and indicators for diabetes classification.
    The target variable has 3 classes (0, 1, 2) representing different diabetes outcomes.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([45])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([3])

    def __init__(self, root: str = "data/diabetes/diabetes.csv"):
        self.root = root
        super().__init__(Path(root))


@final
class MiniBooNEDataset(TabularCSVDataset):
    """
    MiniBooNE dataset wrapped to follow the AFADataset protocol.

    This dataset contains particle physics measurements from the MiniBooNE experiment.
    The target variable has 2 classes (signal and background).

    Rows holding the undocumented -999 placeholder (468 in the UCI file, all
    -999 in every feature) and the four rows whose Feature 20 is off by
    orders of magnitude (-17256.7 to 1.6e7, against [-5.2, 7.5] elsewhere)
    are dropped (issue #129); `get_source_rows` maps instances back to the
    file.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)

    PLACEHOLDER: ClassVar[float] = -999.0
    # Feature 20 is the 20th column; read by position like the base reads
    # the label
    OUTLIER_COLUMN: ClassVar[int] = 19
    OUTLIER_BOUND: ClassVar[float] = 1000.0

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([50])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(self, root: str = "data/miniboone/miniboone.csv"):
        self.root = root
        super().__init__(Path(root))

    @override
    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        features_df, labels = super()._read_source()
        keep = ~(features_df == self.PLACEHOLDER).any(axis=1) & (
            features_df.iloc[:, self.OUTLIER_COLUMN].abs() < self.OUTLIER_BOUND
        )
        return features_df.loc[keep].copy(), labels.loc[keep]


@final
class PhysionetDataset(TabularCSVDataset):
    """
    Physionet dataset wrapped to follow the AFADataset protocol.

    This dataset contains medical measurements from ICU patients.
    The target variable has 2 classes (0, 1) representing different outcomes.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([41])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(self, root: str = "data/physionet/physionet.csv"):
        self.root = root
        super().__init__(Path(root))


class UCIDataset(TabularCSVDataset, ABC):
    """A tabular dataset fetched from the UCI repository into `path` when the file is missing; its feature shape is the file's."""

    config_keys: ClassVar[tuple[str, ...]] = ("path",)

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size(self.features.shape[1:])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(self, path: str):
        self.path: str = path
        super().__init__(Path(path))

    @override
    def _fetch_source(self) -> None:
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self._fetch_uci().to_csv(self.source, index=False)

    @abstractmethod
    def _fetch_uci(self) -> pd.DataFrame:
        """Return the UCI repository's data as the frame `_read_source` reads, with the label last."""


@final
class BankMarketingDataset(UCIDataset):
    @override
    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        df_data = pd.read_csv(self.source, sep=";")
        target_col = "y" if "y" in df_data.columns else "deposit"
        features_df = df_data.drop(columns=[target_col])
        pd.set_option("future.no_silent_downcasting", True)
        labels = cast(
            "pd.Series",
            df_data[target_col].replace({"yes": 1, "no": 0}).astype("int64"),
        )
        # Missing categorical values are a category of their own
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                features_df[col] = LabelEncoder().fit_transform(
                    features_df[col].astype(str)
                )
        return features_df, labels

    @override
    def _fetch_source(self) -> None:
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self._fetch_uci().to_csv(self.source, sep=";", index=False)

    @override
    def _fetch_uci(self) -> pd.DataFrame:
        bank_data = fetch_ucirepo(id=222)
        assert bank_data is not None
        assert bank_data.data is not None
        return pd.concat(
            [bank_data.data.features, bank_data.data.targets],
            axis=1,
        )


@final
class CKDDataset(UCIDataset):
    @override
    def _fetch_uci(self) -> pd.DataFrame:
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
        return df_data


@final
class ACTG175Dataset(UCIDataset):
    @override
    def _fetch_uci(self) -> pd.DataFrame:
        actg_data = fetch_ucirepo(id=890)
        assert actg_data is not None
        assert actg_data.data is not None
        features_df = actg_data.data.features.copy()
        target_df = actg_data.data.targets.copy()
        target_series = target_df.iloc[:, 0].astype(int)
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        return df_data


@final
class ImagenetteDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Imagenette dataset from the FastAI image classification benchmark.

    A subset of 10 easily classified classes from Imagenet.

    Only file lists are kept in memory. The generation indices double as
    positions in `samples`, which lists the `load_subdirs` folders in order,
    so dataset generation loads `train/` then `val/` for every split.
    """

    IMAGENETTE_URL: ClassVar[str] = (
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        subset = copy.deepcopy(self)
        subset.generation_indices = self.generation_indices[
            list(indices)
        ].clone()
        return subset

    @override
    def get_generation_indices(self) -> Tensor:
        return self.generation_indices

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
        data_root: str = "data/imagenette",
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
        self.generation_indices = torch.arange(
            len(self.samples), dtype=torch.long
        )

    def _resolve_index(self, i: int) -> int:
        if i < 0 or i >= int(self.generation_indices.numel()):
            msg = f"Index {i} out of range for dataset of length {len(self)}"
            raise IndexError(msg)
        return int(self.generation_indices[i].item())

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
        return int(self.generation_indices.numel())

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
        """Save only the generation indices and the dataset config, to reconstruct later from raw files on load."""
        torch.save(
            {
                "generation_indices": self.generation_indices,
                "feature_costs": self.get_feature_acquisition_costs(),
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
        generation_indices = load_generation_indices(
            data, path / "dataset.pt", None
        )
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
        obj.generation_indices = generation_indices
        n = len(obj.samples)
        if int(generation_indices.numel()) > 0 and (
            int(generation_indices.min().item()) < 0
            or int(generation_indices.max().item()) >= n
        ):
            msg = (
                f"Loaded generation indices out of bounds: valid [0, {n - 1}]"
            )
            raise ValueError(msg)
        obj.feature_costs = load_saved_feature_costs(
            data, path / "dataset.pt", obj.feature_shape
        )

        return obj


@final
class SyntheticMNISTDataset(InMemoryAFADataset):
    """
    Synthetic MNIST-like dataset with the same shape as MNIST (28x28 = 784 features, 10 classes).

    Generates synthetic image-like data with patterns that can be learned.
    The left half (14 pixels) contains only noise, while the right half contains
    class-specific patterns for label identification.

    Implements the AFADataset protocol.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "noise_std",
        "pattern_intensity",
    )

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

    def __init__(
        self,
        seed: int = 123,
        # Memory-friendly default (60k samples ≈ 200MB RAM)
        n_samples: int = 10000,
        noise_std: float = 0.1,
        pattern_intensity: float = 0.8,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.pattern_intensity = pattern_intensity
        super().__init__()

    @override
    def _build(self) -> tuple[Features, Label]:  # noqa: C901, PLR0915, PLR0912
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Generate labels uniformly
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=rng,
        )

        # Initialize features with noise in image format (C, H, W)
        features = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=(self.n_samples, 1, 28, 28),
            generator=rng,
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
            img = features[i, 0]  # Get the single channel (28, 28)

            # Ensure left half (0:14) remains pure noise - regenerate it
            img[:, 0:14] = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=(28, 14),
                generator=rng,
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
            features[i, 0] = img

        # Normalize features to [0, 1] range like MNIST
        features = torch.clamp(features, 0, 1)
        return features, _one_hot(y_int, self.label_shape[0])
