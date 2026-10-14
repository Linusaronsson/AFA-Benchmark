from pathlib import Path
from typing import ClassVar, final, override

import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


@final
class DiabetesDataset(TabularCSVDataset):
    """
    NHANES participants classified as normal, prediabetic or diabetic from 45 features.

    The source is the CSV of the GDFS repository, which holds Kachuee et
    al.'s preprocessing of NHANES 1999-2016; nothing in this repository
    rebuilds it, so it has no source preparation.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root",)
    # datasets/diabetes.csv of iancovert/dynamic-selection at e2b6f74
    SHA256: ClassVar[str] = (
        "0a7cb293c8819907a44f993e13e62fa00e93457ee7ccfab36ebe761b66df6c3b"
    )

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([45])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([3])

    def __init__(
        self,
        root: str = "data/diabetes/diabetes.csv",
        sha256: str | None = None,
    ):
        self.root = root
        super().__init__(Path(root), sha256)
