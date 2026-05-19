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
from afabench.eval.eval import AFAStepResult, single_afa_step


def assert_tensor_matches(
    tensor: torch.Tensor, expected_tensor: torch.Tensor, desc: str = "tensor"
) -> None:
    assert torch.allclose(tensor, expected_tensor), (
        f"Expected {desc} {expected_tensor.tolist()}, but got {tensor.tolist()}"
    )


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
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
        dtype=torch.float32,
    )


@pytest.fixture
def label() -> torch.Tensor:
    return torch.tensor([0, 1, 1], dtype=torch.int64)


@pytest.fixture
def masked_features() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )


@pytest.fixture
def feature_mask() -> torch.Tensor:
    return torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )


@pytest.fixture
def selection_mask() -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype=torch.bool,
    )


def test_feature_acquisition(
    features: torch.Tensor,
    label: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    selection_mask: torch.Tensor,
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

    result: AFAStepResult = single_afa_step(
        features=features,
        label=label,
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    # Expected: both samples selected action 3, which unmaskes features 4-5
    expected_action = torch.tensor([[3], [3], [3]], dtype=torch.int64)
    assert torch.allclose(result.action, expected_action), (
        f"Expected action {expected_action.tolist()}, but got {result.action.tolist()}"
    )

    expected_masked_features = torch.tensor(
        [[1, 2, 0, 0, 5, 6], [0, 0, 9, 10, 11, 12], [0, 0, 0, 0, 17, 18]],
        dtype=torch.float32,
    )
    assert torch.allclose(result.masked_features, expected_masked_features), (
        f"Expected masked features {expected_masked_features.tolist()}, "
        f"but got {result.masked_features.tolist()}"
    )

    expected_feature_mask = torch.tensor(
        [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.allclose(
        result.feature_mask.float(), expected_feature_mask.float()
    ), (
        f"Expected feature mask {expected_feature_mask.tolist()}, "
        f"but got {result.feature_mask.tolist()}"
    )


def test_stop_selection(
    features: torch.Tensor,
    label: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    selection_mask: torch.Tensor,
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

    result: AFAStepResult = single_afa_step(
        features=features,
        label=label,
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    # Expected: sample 0 stops (action 0), sample 1&2 select action 3
    expected_action = torch.tensor([[0], [3], [3]], dtype=torch.int64)
    assert torch.allclose(result.action, expected_action), (
        f"Expected action {expected_action.tolist()}, but got {result.action.tolist()}"
    )

    expected_masked_features = torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 11, 12], [0, 0, 0, 0, 17, 18]],
        dtype=torch.float32,
    )
    assert torch.allclose(result.masked_features, expected_masked_features), (
        f"Expected masked features {expected_masked_features.tolist()}, "
        f"but got {result.masked_features.tolist()}"
    )

    expected_feature_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.allclose(
        result.feature_mask.float(), expected_feature_mask.float()
    ), (
        f"Expected feature mask {expected_feature_mask.tolist()}, "
        f"but got {result.feature_mask.tolist()}"
    )


def test_force_acquisition(
    features: torch.Tensor,
    label: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    selection_mask: torch.Tensor,
) -> None:
    """Test that single_afa_step forces acquisition when told to. The first non-observed features is chosen."""

    def afa_action_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        # Stop if at least one selection has been made so far, otherwise choose action 3
        assert selection_mask is not None
        should_stop_mask = selection_mask.any(dim=1)
        batch_size = masked_features.shape[0]
        actions = 3 * torch.ones((batch_size, 1), dtype=torch.int64)
        actions[should_stop_mask] = 0
        return actions

    result: AFAStepResult = single_afa_step(
        features=features,
        label=label,
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
        force_acquisition=True,
    )

    expected_action = torch.tensor([[2], [1], [3]], dtype=torch.int64)
    assert_tensor_matches(result.action, expected_action, "action")

    expected_masked_features = torch.tensor(
        [[1, 2, 3, 4, 0, 0], [7, 8, 9, 10, 0, 0], [0, 0, 0, 0, 17, 18]],
        dtype=torch.float32,
    )
    assert_tensor_matches(
        result.masked_features, expected_masked_features, "masked features"
    )

    expected_feature_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
        dtype=torch.bool,
    )
    assert_tensor_matches(
        result.feature_mask.float(),
        expected_feature_mask.float(),
        "feature mask",
    )

    assert_tensor_matches(
        result.acquisition_forced.float(),
        torch.tensor([True, True, False]).float(),
        "acquisition forced mask",
    )
