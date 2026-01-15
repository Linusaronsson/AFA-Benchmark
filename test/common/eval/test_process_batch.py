import pandas as pd
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
from afabench.eval.eval import process_batch


def random_afa_action_fn(
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


def get_deterministic_action(  # noqa: PLR0911
    sample_idx: int, selection_mask: torch.Tensor
) -> int:
    if sample_idx == 0:
        if (selection_mask == torch.tensor([0, 0, 0, 0])).all():
            return 3
        if (selection_mask == torch.tensor([0, 0, 1, 0])).all():
            return 1
        if (selection_mask == torch.tensor([1, 0, 1, 0])).all():
            return 2
        if (selection_mask == torch.tensor([1, 1, 1, 0])).all():
            return 0
    elif sample_idx == 1:
        if (selection_mask == torch.tensor([0, 0, 0, 0])).all():
            return 2
        if (selection_mask == torch.tensor([0, 1, 0, 0])).all():
            return 4
        if (selection_mask == torch.tensor([0, 1, 0, 1])).all():
            return 0
    msg = "Not reachable"
    raise RuntimeError(msg)


def deterministic_afa_action_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,  # noqa: ARG001
    selection_mask: SelectionMask | None = None,
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> AFAAction:
    assert selection_mask is not None
    batch_size = masked_features.shape[0]
    actions = torch.zeros(batch_size, 1, dtype=torch.int)
    for sample_idx in range(batch_size):
        actions[sample_idx] = get_deterministic_action(
            sample_idx, selection_mask[sample_idx]
        )
    return actions


def get_deterministic_builtin_prediction(
    sample_idx: int, masked_features: torch.Tensor
) -> int:
    if sample_idx == 0:
        if torch.allclose(
            masked_features,
            torch.tensor([1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]),
        ):
            return 1
        if torch.allclose(
            masked_features,
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]),
        ):
            return 0
    elif sample_idx == 1:
        if torch.allclose(
            masked_features,
            torch.tensor([0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        ):
            return 1
        if torch.allclose(
            masked_features,
            torch.tensor([0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 15.0, 16.0]),
        ):
            return 2
    msg = "Not reachable"
    raise RuntimeError(msg)


def get_deterministic_external_prediction(
    sample_idx: int, masked_features: torch.Tensor
) -> int:
    print(f"{sample_idx=}, {masked_features=}")
    if sample_idx == 0:
        if torch.allclose(
            masked_features,
            torch.tensor([1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]),
        ):
            return 2
        if torch.allclose(
            masked_features,
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]),
        ):
            return 1
    elif sample_idx == 1:
        if torch.allclose(
            masked_features,
            torch.tensor([0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        ):
            return 2
        if torch.allclose(
            masked_features,
            torch.tensor([0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 15.0, 16.0]),
        ):
            return 2
    msg = "Not reachable"
    raise RuntimeError(msg)


def deterministic_builtin_afa_predict_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> Label:
    batch_size = masked_features.shape[0]
    predictions = torch.zeros((batch_size, 3), dtype=torch.float32)
    for sample_idx in range(batch_size):
        class_pred = get_deterministic_builtin_prediction(
            sample_idx, masked_features[sample_idx]
        )
        predictions[sample_idx, class_pred] = 1.0
    return predictions


def deterministic_external_afa_predict_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> Label:
    batch_size = masked_features.shape[0]
    predictions = torch.zeros((batch_size, 3), dtype=torch.float32)
    for sample_idx in range(batch_size):
        class_pred = get_deterministic_external_prediction(
            sample_idx, masked_features[sample_idx]
        )
        predictions[sample_idx, class_pred] = 1.0
    return predictions


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


@pytest.fixture
def features() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.float32,
    )


@pytest.fixture
def feature_mask() -> torch.Tensor:
    return torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )


@pytest.fixture
def masked_features() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 0, 0, 0, 0, 0, 0], [0, 0, 11, 12, 0, 0, 0, 0]],
        dtype=torch.float32,
    )


@pytest.fixture
def true_label() -> torch.Tensor:
    return torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)


def create_expected_eval_df(
    actions: list[int],
    builtin_preds: list[int],
    external_preds: list[int],
    true_class: int,
) -> pd.DataFrame:
    assert len(actions) == len(builtin_preds) == len(external_preds)
    # The stop actions must always come last
    if 0 in actions:
        assert 0 not in actions[:-1]
    rows = []
    prev_selections = []
    for action, builtin_pred, external_pred in zip(
        actions, builtin_preds, external_preds, strict=True
    ):
        rows.append(
            {
                "prev_selections_performed": prev_selections.copy(),
                "action_performed": action,
                "builtin_predicted_class": builtin_pred,
                "external_predicted_class": external_pred,
                "true_class": true_class,
            }
        )
        prev_selections.append(action - 1)
    return pd.DataFrame(rows)


def is_column_subset_row(row: pd.Series, expected_row: pd.Series) -> bool:
    """
    Check whether row is a subset of expected_row column-wise.

    This means that expected_row can contain columns that are not in row.
    """
    for key, value in row.items():
        if not (key in expected_row and expected_row[key] == value):
            return False
    return True


def is_column_subset_df(df: pd.DataFrame, expected_df: pd.DataFrame) -> bool:
    """
    Check whether df is a subset of expected_df column-wise.

    This means that expected_df can contain columns that are not in df, but they have to have the same number of rows.
    """
    for _, row in df.iterrows():
        row_matched = False
        for _, expected_row in expected_df.iterrows():
            if is_column_subset_row(row, expected_row):
                row_matched = True
                break
        assert row_matched, (
            f"\n{row}\n did not match any row in the expected dataframe \n{expected_df.to_string()}\n"
        )
    return True


def test_batch_dynamics_without_budget(
    features: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    true_label: torch.Tensor,
) -> None:
    """Test that process_batch correctly simulates feature acquisition with deterministic actions and predictions."""
    df = process_batch(
        afa_action_fn=deterministic_afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=masked_features,
        initial_feature_mask=feature_mask,
        true_label=true_label,
        external_afa_predict_fn=deterministic_external_afa_predict_fn,
        builtin_afa_predict_fn=deterministic_builtin_afa_predict_fn,
        selection_budget=None,
    )

    # Quick check that dataframe at least contains the correct column names
    assert "prev_selections_performed" in df
    assert "action_performed" in df
    assert "builtin_predicted_class" in df
    assert "external_predicted_class" in df
    assert "true_class" in df

    sample0_actions = [3, 1, 2, 0]
    sample0_builtin_preds = [1, 1, 0, 0]
    sample0_external_preds = [2, 2, 1, 1]

    sample1_actions = [2, 4, 0]
    sample1_builtin_preds = [1, 2, 2]
    sample1_external_preds = [2, 2, 2]

    expected_rows_sample0 = create_expected_eval_df(
        sample0_actions,
        sample0_builtin_preds,
        sample0_external_preds,
        true_class=0,
    )
    expected_rows_sample1 = create_expected_eval_df(
        sample1_actions,
        sample1_builtin_preds,
        sample1_external_preds,
        true_class=1,
    )

    expected_df = pd.concat([expected_rows_sample0, expected_rows_sample1])

    # Check total number of rows
    assert len(df) == len(expected_df), (
        f"Expected {len(expected_df)} rows in DataFrame, got {len(df)}.\n"
        f"Actual DataFrame:\n{df.to_string()}"
    )

    assert is_column_subset_df(df, expected_df), (
        f"DataFrame does not contain expected rows.\n"
        f"Expected rows to match:\n{expected_df}\n\n"
        f"Actual DataFrame:\n{df.to_string()}"
    )


def test_process_batch_respects_budget(
    features: torch.Tensor,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    true_label: torch.Tensor,
) -> None:
    """Test that process_batch runs and does not include results that are incompatible with the budget given."""
    df_batch = process_batch(
        afa_action_fn=random_afa_action_fn,
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
