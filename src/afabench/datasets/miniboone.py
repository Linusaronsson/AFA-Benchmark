from pathlib import Path
from typing import ClassVar, final, override

import pandas as pd
import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


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
        return self.drop_invalid_rows(*super()._read_source())

    @classmethod
    def drop_invalid_rows(
        cls, features_df: pd.DataFrame, labels: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Drop the placeholder and outlier rows of MiniBooNE's source frames."""
        keep = ~(features_df == cls.PLACEHOLDER).any(axis=1) & (
            features_df.iloc[:, cls.OUTLIER_COLUMN].abs() < cls.OUTLIER_BOUND
        )
        return features_df.loc[keep].copy(), labels.loc[keep]
