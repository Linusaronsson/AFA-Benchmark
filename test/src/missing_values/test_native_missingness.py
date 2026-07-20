"""
Datasets that arrive incomplete must keep that missingness, not impute it away.

Easy to lose silently: the loaders mean-fill, and a mean-filled column looks
complete. A run would then present itself as a native-missingness arm while
measuring nothing.
"""

import pytest
import torch

from afabench.datasets.datasets import CKDDataset, PhysionetDataset
from afabench.missing_values.views import native_observed_mask


class _Complete:
    """A dataset with nothing missing, like diabetes or miniboone."""


class _Incomplete:
    def __init__(self, mask: torch.Tensor):
        self.native_observed_mask = mask


def test_native_mask_is_returned_as_bool() -> None:
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.uint8)

    got = native_observed_mask(_Incomplete(mask))

    assert got.dtype is torch.bool
    assert torch.equal(
        got, torch.tensor([[True, False, True], [True, True, False]])
    )


def test_complete_dataset_is_rejected_by_name() -> None:
    """Returning an all-ones mask here would silently measure nothing."""
    with pytest.raises(ValueError, match=r"_Complete.*native"):
        native_observed_mask(_Complete())


@pytest.mark.parametrize(
    ("dataset_cls", "kwargs"),
    [
        (PhysionetDataset, {}),
        (CKDDataset, {"path": "extra/data/misc/chronic_kidney_disease.csv"}),
    ],
    ids=["physionet", "ckd"],
)
def test_incomplete_datasets_expose_a_mask(
    dataset_cls: type, kwargs: dict[str, str]
) -> None:
    """Source CSVs are gitignored, so skip rather than fail without them."""
    try:
        dataset = dataset_cls(**kwargs)
    except (FileNotFoundError, OSError):
        pytest.skip(f"{dataset_cls.__name__} source data not available")

    mask = native_observed_mask(dataset)
    features, _ = dataset.get_all_data()

    assert mask.shape == features.shape
    assert not mask.all(), f"{dataset_cls.__name__} reports no missingness"
    assert mask.any(), f"{dataset_cls.__name__} reports nothing observed"
