import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from torch.utils.data.dataloader import default_collate

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset


def default_create_subset[T: AFADataset](
    dataset: T, indices: Sequence[int]
) -> T:
    """Return a subset of a dataset using default logic for in-memory datasets with .features and .labels."""
    subset = copy.deepcopy(dataset)
    indices_list = list(indices)
    if hasattr(subset, "features") and hasattr(subset, "labels"):
        features = getattr(subset, "features")  # noqa: B009
        labels = getattr(subset, "labels")  # noqa: B009
        setattr(subset, "features", features[indices_list])  # noqa: B010
        setattr(subset, "labels", labels[indices_list])  # noqa: B010
        if hasattr(subset, "n_samples"):
            setattr(subset, "n_samples", len(indices))  # noqa: B010
    else:
        msg = "default_create_subset requires 'features' and 'labels' attributes on the dataset."
        raise AttributeError(msg)
    return subset


def flatten_features_collate(n_feature_dims: int) -> Callable:  # pyright: ignore[reportMissingTypeArgument]
    def collate(batch):  # noqa: ANN001, ANN202
        features, labels = default_collate(batch)

        flat_features = features.flatten(start_dim=-n_feature_dims)

        return flat_features, labels

    return collate
