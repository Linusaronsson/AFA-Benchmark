from pathlib import Path
from typing import ClassVar, final, override

import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


@final
class PhysionetChallenge2012Dataset(TabularCSVDataset):
    """
    PhysioNet Challenge 2012 dataset following the AFADataset protocol.

    This dataset contains medical measurements from ICU patients.
    The target is in-hospital mortality (0: survival, 1: death).
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

    def __init__(
        self, root: str = "data/physionet_challenge_2012/physionet.csv"
    ):
        self.root = root
        super().__init__(Path(root))
