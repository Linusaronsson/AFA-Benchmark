from pathlib import Path
from typing import ClassVar, final, override

import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


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
