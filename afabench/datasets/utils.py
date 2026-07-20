import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from torch.utils.data.dataloader import default_collate

if TYPE_CHECKING:
    from afabench.core.types import AFADataset


def default_create_subset[T: AFADataset](
    dataset: T,
    indices: Sequence[int],
) -> T:
    """Create an in-memory dataset subset by slicing sample-aligned attributes."""
    subset = copy.deepcopy(dataset)
    indices_list = list(indices)

    if not hasattr(subset, "features") or not hasattr(subset, "labels"):
        msg = (
            "default_create_subset requires 'features' and 'labels' "
            "attributes on the dataset."
        )
        raise AttributeError(msg)

    features = getattr(subset, "features")  # noqa: B009
    labels = getattr(subset, "labels")  # noqa: B009

    setattr(subset, "features", features[indices_list])  # noqa: B010
    setattr(subset, "labels", labels[indices_list])  # noqa: B010

    if hasattr(subset, "native_observed_mask"):
        native_observed_mask = getattr(  # noqa: B009
            subset,
            "native_observed_mask",
        )
        if native_observed_mask is not None:
            setattr(  # noqa: B010
                subset,
                "native_observed_mask",
                native_observed_mask[indices_list],
            )

    if hasattr(subset, "n_samples"):
        setattr(subset, "n_samples", len(indices_list))  # noqa: B010

    return subset


def flatten_features_collate(n_feature_dims: int) -> Callable:  # pyright: ignore[reportMissingTypeArgument]
    def collate(batch):  # noqa: ANN001, ANN202
        features, labels, *row_extras = default_collate(batch)

        flat_features = features.flatten(start_dim=-n_feature_dims)
        flat_extras = [
            extra.flatten(start_dim=-n_feature_dims)
            if isinstance(extra, torch.Tensor)
            else extra
            for extra in row_extras
        ]
        return flat_features, labels, *flat_extras

    return collate
