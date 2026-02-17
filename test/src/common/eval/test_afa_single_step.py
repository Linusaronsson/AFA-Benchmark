import pytest
import torch

from afabench.common.custom_types import (
    AFAAction,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.eval.eval import single_afa_step


def afa_unmask_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    features: Features,  # noqa: ARG001
    afa_selection: AFASelection,
    selection_mask: SelectionMask,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> FeatureMask:
    # 6 features but selection is 0-2 (0-based). Unmask a "block" of features.
    batch_size, num_features = masked_features.shape
    new_feature_mask = feature_mask.clone()
    for i in range(batch_size):
        selection = int(afa_selection[i].item())
        start_idx = selection * 2
        end_idx = min(start_idx + 2, num_features)
        new_feature_mask[i, start_idx:end_idx] = 1

    return new_feature_mask


@pytest.fixture
def features() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        dtype=torch.float32,
    )


@pytest.fixture
def label() -> torch.Tensor:
    return torch.tensor([0, 1], dtype=torch.int64)


@pytest.fixture
def masked_features() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 0, 0]],
        dtype=torch.float32,
    )


@pytest.fixture
def feature_mask() -> torch.Tensor:
    return torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]],
        dtype=torch.bool,
    )


def test_single_afa_step(
    features: torch.Tensor,
    label: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
) -> None:
    """Test that single_afa_step correctly updates features when all actions are non-zero."""

    def afa_action_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        # Always output action 3 (selection index 2, unmasking features 4-5)
        return 3 * torch.ones((masked_features.shape[0], 1), dtype=torch.int64)

    action, new_masked_features, new_feature_mask, _, _ = single_afa_step(
        features=features,
        label=label,
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=feature_mask,
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    # Expected: both samples selected action 3, which unmaskes features 4-5
    expected_action = torch.tensor([[3], [3]], dtype=torch.int64)
    assert torch.allclose(
        action, expected_action
    ), f"Expected action {expected_action.tolist()}, but got {action.tolist()}"

    expected_masked_features = torch.tensor(
        [[1, 2, 0, 0, 5, 6], [0, 0, 9, 10, 11, 12]],
        dtype=torch.float32,
    )
    assert torch.allclose(new_masked_features, expected_masked_features), (
        f"Expected masked features {expected_masked_features.tolist()}, "
        f"but got {new_masked_features.tolist()}"
    )

    expected_feature_mask = torch.tensor(
        [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.allclose(
        new_feature_mask.float(), expected_feature_mask.float()
    ), (
        f"Expected feature mask {expected_feature_mask.tolist()}, "
        f"but got {new_feature_mask.tolist()}"
    )


def test_single_afa_step_stop_selection(
    features: torch.Tensor,
    label: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
) -> None:
    """Test that single_afa_step correctly handles mixed actions (some samples stop, others continue)."""

    def afa_action_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        # Stop if first feature is observed (sample 0), otherwise select action 3
        batch_size = masked_features.shape[0]
        actions = 3 * torch.ones((batch_size, 1), dtype=torch.int64)
        for i in range(batch_size):
            if masked_features[i, 0] == 1:
                actions[i] = 0
        return actions

    action, new_masked_features, new_feature_mask, _, _ = single_afa_step(
        features=features,
        label=label,
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=feature_mask,
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    # Expected: sample 0 stops (action 0), sample 1 selects action 3
    expected_action = torch.tensor([[0], [3]], dtype=torch.int64)
    assert torch.allclose(
        action, expected_action
    ), f"Expected action {expected_action.tolist()}, but got {action.tolist()}"

    expected_masked_features = torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 11, 12]],
        dtype=torch.float32,
    )
    assert torch.allclose(new_masked_features, expected_masked_features), (
        f"Expected masked features {expected_masked_features.tolist()}, "
        f"but got {new_masked_features.tolist()}"
    )

    expected_feature_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.allclose(
        new_feature_mask.float(), expected_feature_mask.float()
    ), (
        f"Expected feature mask {expected_feature_mask.tolist()}, "
        f"but got {new_feature_mask.tolist()}"
    )
