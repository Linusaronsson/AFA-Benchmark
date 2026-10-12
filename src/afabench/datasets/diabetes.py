from pathlib import Path
from typing import ClassVar, final, override

import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


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
