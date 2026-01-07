from collections.abc import Sequence
from pathlib import Path
from typing import Self, override

import torch

from afabench.common.custom_types import AFADataset, Features, Label


class ExtendedAFADataset(AFADataset):
    """Extend an AFADataset with additional features and labels."""

    def __init__(
        self,
        base_dataset: AFADataset,
        additional_features: Features,
        additional_labels: Label,
    ):
        # Assume that both features and labels have a single batch dimension
        assert additional_features.shape[1:] == base_dataset.feature_shape
        assert additional_labels.shape[1:] == base_dataset.label_shape
        # As many new features as labels
        assert additional_features.shape[0] == additional_labels.shape[0]

        self.base_dataset: AFADataset = base_dataset
        self.additional_features: Features = additional_features
        self.additional_labels: Label = additional_labels

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return self.base_dataset.feature_shape

    @property
    @override
    def label_shape(self) -> torch.Size:
        return self.base_dataset.label_shape

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        new_idx = idx - len(self.base_dataset)
        return self.additional_features[new_idx], self.additional_labels[
            new_idx
        ]

    @override
    def __len__(self) -> int:
        return len(self.base_dataset) + self.additional_features.shape[0]

    @override
    def get_all_data(
        self,
    ) -> tuple[Features, Label]:
        base_features, base_labels = self.base_dataset.get_all_data()
        all_features = torch.concat(
            [base_features, self.additional_features], dim=0
        )
        all_labels = torch.concat([base_labels, self.additional_labels], dim=0)
        return all_features, all_labels

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        raise NotImplementedError

    @override
    def get_feature_acquisition_costs(self) -> torch.Tensor:
        return self.base_dataset.get_feature_acquisition_costs()
