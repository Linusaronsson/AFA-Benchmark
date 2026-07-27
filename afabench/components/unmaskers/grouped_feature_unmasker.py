from pathlib import Path
from typing import final, override

import pandas as pd
import torch

from afabench.core.types import (
    AFASelection,
    AFAUnmasker,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class GroupedFeatureUnmasker(AFAUnmasker):
    """Reveal flat tabular features in fixed, atomic acquisition groups."""

    def __init__(
        self,
        group_ids: list[int] | None = None,
        groups_path: str | None = None,
    ) -> None:
        if (group_ids is None) == (groups_path is None):
            msg = "Provide exactly one of group_ids or groups_path."
            raise ValueError(msg)
        if groups_path is not None:
            frame = pd.read_csv(Path(groups_path))
            required = {"feature_index", "group_id"}
            if not required.issubset(frame.columns):
                msg = f"{groups_path} must contain columns {sorted(required)}."
                raise ValueError(msg)
            frame = frame.sort_values("feature_index")
            expected = list(range(len(frame)))
            if frame["feature_index"].tolist() != expected:
                msg = "feature_index must be contiguous and zero-based."
                raise ValueError(msg)
            group_ids = frame["group_id"].astype(int).tolist()
        assert group_ids is not None
        raw = torch.tensor(group_ids, dtype=torch.long)
        if raw.ndim != 1 or raw.numel() == 0:
            msg = "group_ids must be a nonempty one-dimensional list."
            raise ValueError(msg)
        if (raw < 0).any():
            msg = "group_ids must be nonnegative."
            raise ValueError(msg)
        _, self.group_ids = torch.unique(
            raw,
            sorted=True,
            return_inverse=True,
        )
        self.n_groups = int(self.group_ids.max().item()) + 1

    @override
    def set_seed(self, seed: int | None) -> None:
        pass

    def _validate_feature_shape(self, feature_shape: torch.Size) -> None:
        if len(feature_shape) != 1:
            msg = "GroupedFeatureUnmasker requires flat tabular features."
            raise ValueError(msg)
        if feature_shape.numel() != len(self.group_ids):
            msg = (
                f"Expected {len(self.group_ids)} features, got "
                f"{feature_shape.numel()}."
            )
            raise ValueError(msg)

    @override
    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        self._validate_feature_shape(feature_costs.shape)
        costs = torch.zeros(
            self.n_groups,
            dtype=feature_costs.dtype,
            device=feature_costs.device,
        )
        return costs.scatter_add(
            0,
            self.group_ids.to(feature_costs.device),
            feature_costs,
        )

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        self._validate_feature_shape(feature_shape)
        return self.n_groups

    @override
    def feature_availability_to_selection_availability(
        self,
        feature_availability: FeatureMask,
    ) -> SelectionMask:
        if feature_availability.ndim != 2:
            msg = "Expected feature availability with shape (batch, feature)."
            raise ValueError(msg)
        self._validate_feature_shape(feature_availability.shape[1:])
        group_ids = self.group_ids.to(feature_availability.device)
        return torch.stack(
            [
                feature_availability[:, group_ids == group].all(dim=1)
                for group in range(self.n_groups)
            ],
            dim=1,
        )

    @override
    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        del masked_features, features, selection_mask, label
        if feature_mask.ndim != 2:
            msg = "Expected feature mask with shape (batch, feature)."
            raise ValueError(msg)
        resolved_shape = feature_shape or feature_mask.shape[1:]
        self._validate_feature_shape(resolved_shape)
        selections = afa_selection.reshape(-1)
        if ((selections < 0) | (selections >= self.n_groups)).any():
            msg = "Selection index is outside the configured groups."
            raise ValueError(msg)
        selected_features = (
            self.group_ids.to(feature_mask.device)[None, :]
            == selections[:, None]
        )
        return feature_mask | selected_features
