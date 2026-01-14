import torch

from afabench.common.afa_methods import SequentialDummyAFAMethod
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.eval.eval import process_batch


def test_sequential_dummy_method_never_selects_0() -> None:
    """Test that setting prob_select_0=0 in SequentialDummyAFAMethod never performs the 0 selection."""
    method = SequentialDummyAFAMethod(
        n_classes=2,
        prob_select_0=0.0,
        device=torch.device("cpu"),
    )

    features = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.float32,
    )
    masked_features = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    feature_mask = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    true_label = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    def afa_unmask_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,  # noqa: ARG001
        afa_selection: AFASelection,
        selection_mask: SelectionMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> FeatureMask:
        # 8 features but selection is 0-3 (0-based). Unmask a "block" of features.
        batch_size, num_features = masked_features.shape
        new_feature_mask = feature_mask.clone()
        for i in range(batch_size):
            selection = int(afa_selection[i].item())
            if selection >= 0:
                start_idx = selection * 2
                end_idx = min(start_idx + 2, num_features)
                new_feature_mask[i, start_idx:end_idx] = 1

        return new_feature_mask

    df_batch = process_batch(
        afa_action_fn=method.act,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=masked_features,
        initial_feature_mask=feature_mask,
        true_label=true_label,
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=None,
        selection_budget=None,
    )

    # Check that we get at least some selections since prob_select_0=0.0
    # We should have more than just 2 rows (initial selections) since we never select 0
    assert len(df_batch) > 2, (
        f"Expected more than 2 rows when prob_select_0=0.0, got {len(df_batch)}"
    )

    # Verify that all rows have at least one selection (selections_performed > 0)
    for _, row in df_batch.iterrows():
        selections_performed = row["selections_performed"]
        assert selections_performed > 0, (
            f"Expected selections_performed > 0 when prob_select_0=0.0, but got {selections_performed}"
        )

        # Furthermore, the selections should be made in order. So "prev_selections_performed" should be an ordered list with no gaps, and "selections_performed" should be the count of selections made.
        prev_selections_performed = row["prev_selections_performed"]
        assert prev_selections_performed == sorted(
            prev_selections_performed
        ), (
            f"Expected prev_selections_performed {prev_selections_performed} to be sorted."
        )
        # The selections_performed count should match the length of prev_selections_performed plus 1 (for the current selection)
        expected_count = len(prev_selections_performed) + 1
        assert selections_performed == expected_count, (
            f"Expected selections_performed {selections_performed} to be len(prev_selections_performed) + 1 = {expected_count} given prev_selections_performed {prev_selections_performed}."
        )


def test_sequential_dummy_method_always_selects_0() -> None:
    """Test that setting prob_select_0=1 in SequentialDummyAFAMethod always performs the 0 selection."""
    method = SequentialDummyAFAMethod(
        n_classes=2,
        prob_select_0=1.0,
        device=torch.device("cpu"),
    )

    features = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.float32,
    )
    masked_features = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    feature_mask = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    true_label = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    def afa_unmask_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,  # noqa: ARG001
        afa_selection: AFASelection,
        selection_mask: SelectionMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> FeatureMask:
        # 8 features but selection is 0-3 (0-based). Unmask a "block" of features.
        batch_size, num_features = masked_features.shape
        new_feature_mask = feature_mask.clone()
        for i in range(batch_size):
            selection = int(afa_selection[i].item())
            if selection >= 0:
                start_idx = selection * 2
                end_idx = min(start_idx + 2, num_features)
                new_feature_mask[i, start_idx:end_idx] = 1

        return new_feature_mask

    df_batch = process_batch(
        afa_action_fn=method.act,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=masked_features,
        initial_feature_mask=feature_mask,
        true_label=true_label,
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=None,
        selection_budget=None,
    )

    # We should only have two rows since we always select 0 (stop) immediately
    assert len(df_batch) == 2, (
        f"Expected 2 rows in the result DataFrame, got {len(df_batch)}"
    )
    # All selections_performed should be 0 (no selections since we always stop immediately)
    for _, row in df_batch.iterrows():
        selections_performed = row["selections_performed"]
        assert selections_performed == 0, (
            f"Expected selections_performed == 0 (stop) when prob_select_0=1.0, but got {selections_performed}"
        )
