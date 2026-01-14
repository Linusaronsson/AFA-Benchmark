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
from afabench.eval.eval import process_batch


def test_process_batch_respects_budget() -> None:
    """Test that process_batch runs and does not include results that are incompatible with the budget given."""
    features = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.float32,
    )
    masked_features = torch.tensor(
        [[1, 2, 0, 0, 0, 0, 0, 0], [0, 0, 9, 10, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    feature_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    true_label = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    def afa_action_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        # Random selection (not 0)
        return torch.randint(
            1, 5, (masked_features.shape[0], 1), dtype=torch.int64
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
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=masked_features,
        initial_feature_mask=feature_mask,
        true_label=true_label,
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=None,
        selection_budget=2,
    )

    # We expect 4 rows, since each sample gets 2 selections
    assert len(df_batch) == 4, (
        f"Expected 4 rows in the result DataFrame, got {len(df_batch)}."
    )
