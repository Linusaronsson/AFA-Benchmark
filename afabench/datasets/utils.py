import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afabench.core.types import AFADataset


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
