import hashlib
from abc import ABC
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self, cast, override
from urllib.request import urlopen

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from afabench.core.types import Features, Label
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot


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


class SourceChecksumError(ValueError):
    """A source file is not the one its dataset is defined on."""


class NoSourcePreparationError(NotImplementedError):
    """The dataset's prepared source cannot be rebuilt from its provider's files."""


def download(url: str) -> bytes:
    """Return the content at `url`."""
    with urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def check_sha256(
    content: bytes, sha256: str, origin: object, hint: str = ""
) -> None:
    """Raise `SourceChecksumError` unless `content`, read from `origin`, has SHA-256 `sha256`."""
    actual = hashlib.sha256(content).hexdigest()
    if actual != sha256:
        msg = f"{origin} has SHA-256 {actual}, expected {sha256}{hint}"
        raise SourceChecksumError(msg)


class TabularCSVDataset(InMemoryAFADataset, ABC):
    """
    A real-world tabular dataset read from one CSV file, in the file's row order (ADR 0004).

    The file is the dataset's prepared source (ADR 0014), checked against
    `SHA256` on every generation. It is committed to the repository unless
    `SOURCE_COMMITTED` is false, when generation prepares it if missing.
    `make_prepared_source` is where a subclass makes it from its
    provider's raw files.

    By default the last column holds the class label and the other columns
    the features; `_read_source` is where a subclass reads its file
    differently. Categorical feature columns are label-encoded, missing
    values are filled with the column mean, and every feature is
    z-normalized over the rows kept. The feature names and each instance's
    source row are saved in dataset.pt beside the tensors.
    """

    # SHA-256 of the prepared source the dataset is defined on
    SHA256: ClassVar[str]
    # False when the source's license does not permit committing it
    SOURCE_COMMITTED: ClassVar[bool] = True
    # The constructor argument holding the source's path
    source_config_key: ClassVar[str] = "root"

    feature_names: list[str]
    # Row of each instance in the source file, counting data rows from 0;
    # differs from the generation index once `_read_source` drops rows
    source_rows: Tensor

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(self, source: Path, sha256: str | None = None):
        # Used only to build the data; the subclass's constructor arguments
        # they come from are the config
        self.source: Path = source
        self.source_sha256: str = sha256 or self.SHA256
        super().__init__()

    @classmethod
    def prepare_source(cls, source: Path, sha256: str | None = None) -> None:
        """
        Write the prepared source to `source`, made from the provider's raw files.

        It must have SHA-256 `sha256`, by default `SHA256`; a file with
        another is not written, so a changed provider file is not kept.
        """
        content = cls.make_prepared_source()
        check_sha256(
            content, sha256 or cls.SHA256, f"Prepared {cls.__name__} source"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        partial = source.with_name(source.name + ".partial")
        partial.write_bytes(content)
        partial.replace(source)

    @classmethod
    def make_prepared_source(cls) -> bytes:
        """Return the prepared source, made from the provider's raw files, without checking it."""
        msg = (
            f"{cls.__name__} has no source preparation; its source is "
            "committed to the repository"
        )
        raise NoSourcePreparationError(msg)

    def _check_source(self) -> None:
        if not self.source.exists():
            if not self.SOURCE_COMMITTED:
                self.prepare_source(self.source, self.source_sha256)
                return
            msg = (
                f"{type(self).__name__} source not found at {self.source}; "
                "it is committed to the repository, or prepare it with "
                "`uv run python scripts/dataset_generation/prepare_source.py "
                "<dataset key>`"
            )
            raise FileNotFoundError(msg)
        check_sha256(
            self.source.read_bytes(),
            self.source_sha256,
            f"{type(self).__name__} source {self.source}",
            "; restore the committed file"
            if self.SOURCE_COMMITTED
            else "; remove it to prepare it again",
        )

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
        self._check_source()
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
        return features, one_hot(
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
