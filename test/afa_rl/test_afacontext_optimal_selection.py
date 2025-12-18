"""Check that AFAContextSmartMethod works as expected."""

import pytest
import torch

from afabench.afa_rl.common.utils import afacontext_optimal_selection
from afabench.common.datasets.datasets import AFAContextDataset


@pytest.fixture
def setup_data() -> AFAContextDataset:
    data = AFAContextDataset(n_samples=1)
    return data


def test_chooses_first_feature() -> None:
    """Test that the AFAContextSmartMethod always chooses the first feature at the beginning."""
    masked_features = torch.zeros(2, 20)
    feature_mask = torch.zeros(2, 20, dtype=torch.bool)
    selection = afacontext_optimal_selection(masked_features, feature_mask)
    assert torch.equal(selection, torch.zeros(2, dtype=torch.int64)), (
        "Expected selection to be all zeros, but got non-zero values."
    )


def test_chooses_next_three_features() -> None:
    """Test that the AFAContextSmartMethod chooses the next three features based on the context."""
    masked_features = torch.tensor(
        [
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.float32,
    )
    feature_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    for i in range(3):
        selection = afacontext_optimal_selection(masked_features, feature_mask)
        for j, s in enumerate(selection):
            context_start_idx = masked_features[j, 0] * 3 + 1
            assert s == context_start_idx + i, f"Expected selection to be {
                context_start_idx + i
            }, but got {s}"
            feature_mask[j, s] = 1
