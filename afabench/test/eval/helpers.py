from collections.abc import Sequence

import pandas as pd
import torch

from afabench.common.custom_types import (
    AFAAction,
    AFAActionFn,
    AFAUnmaskFn,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.eval.eval import process_batch


def assert_where_selections_there_cost(
    df: pd.DataFrame, selections: Sequence[int], cost: float
) -> None:
    """Assert that wherever "prev_selections_performed" in the dataframe is equal to `selections`, "accumulated_cost" has to be equal to `cost`."""
    rows = df[df["prev_selections_performed"].apply(lambda x: x == selections)]
    assert len(rows) == 1, (
        f"While evaluating dataframe \n{df}\n, expected exactly one row to contain prev_selections_performed={selections}, instead got \n{rows}\n"
    )
    row = rows.iloc[0]
    assert row["accumulated_cost"] == cost


def assert_where_selections_there_action(
    df: pd.DataFrame, selections: Sequence[int], action: int
) -> None:
    """Assert that wherever "prev_selections_performed" in the dataframe is equal to `selections`, "action_performed" has to be equal to `action`."""
    rows = df[df["prev_selections_performed"].apply(lambda x: x == selections)]
    assert len(rows) == 1, (
        f"While evaluating dataframe \n{df}\n, expected exactly one row to contain prev_selections_performed={selections}, instead got \n{rows}\n"
    )
    row = rows.iloc[0]
    assert row["action_performed"] == action


def get_deterministic_afa_action_fn(actions: Sequence[int]) -> AFAActionFn:
    """Return an AFAActionFn that outputs actions in a specified order."""
    action_idx = 0

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        assert masked_features.ndim == 2, (
            "assuming single batch dimension and single feature dimension"
        )
        batch_size = masked_features.shape[0]
        nonlocal action_idx
        action_tensor = (
            torch.ones((batch_size, 1), dtype=torch.long) * actions[action_idx]
        )
        action_idx += 1
        return action_tensor

    return f


def get_batched_deterministic_afa_action_fn(
    actions: Sequence[Sequence[int]],
) -> AFAActionFn:
    """
    Return an AFAActionFn that outputs actions in a specified order.

    Handles variable batch sizes that occur when process_batch removes samples
    that have completed. Tracks which original samples are still active by
    monitoring the batch size.
    """
    action_idx = 0
    original_batch_size = len(actions)
    active_indices = list(range(original_batch_size))

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        assert masked_features.ndim == 2, (
            "assuming single batch dimension and single feature dimension"
        )
        batch_size = masked_features.shape[0]
        nonlocal action_idx, active_indices

        # If batch size changed, remove the sample that finished
        # Samples are removed from process_batch when they complete,
        # and they're always removed from the end of active_indices
        if batch_size < len(active_indices):
            active_indices = active_indices[:batch_size]

        action_tensor = torch.zeros((batch_size, 1), dtype=torch.long)
        for local_idx in range(batch_size):
            global_idx = active_indices[local_idx]
            action_tensor[local_idx, 0] = actions[global_idx][action_idx]
        action_idx += 1
        return action_tensor

    return f


def get_batch_from_costs_and_budget(
    afa_action_fn: AFAActionFn,
    afa_unmask_fn: AFAUnmaskFn,
    selection_budget: float,
    selection_costs: Sequence[float],
    batch_size: int = 2,
    n_features: int = 5,
) -> pd.DataFrame:
    features = torch.zeros((batch_size, n_features))
    true_label = torch.zeros((batch_size, 2))
    true_label[:, :] = torch.tensor([0, 1])
    df = process_batch(
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=torch.zeros_like(features),
        initial_feature_mask=torch.zeros_like(features, dtype=torch.bool),
        true_label=true_label,
        feature_shape=torch.Size((n_features,)),
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=None,
        selection_budget=selection_budget,
        selection_costs=selection_costs,
    )
    return df


def assert_terminated_after_n_steps(
    df: pd.DataFrame, idx: int, n_steps: int
) -> None:
    """
    Assert that sample `idx` terminated after performing `n_steps` actions.

    Args:
        df: The dataframe from process_batch
        idx: The sample index to check
        n_steps: The number of actual selections
    """
    sample_rows = df[df["idx"] == idx]
    assert len(sample_rows) == n_steps, (
        f"Expected {n_steps} rows for sample {idx}, but got {len(sample_rows)}. "
    )
    # There should be a single row where the action = 0 (stop action)
    for _, row in sample_rows.iterrows():
        if len(row["prev_selections_performed"]) == n_steps - 1:
            assert row["action_performed"] == 0
        else:
            assert row["action_performed"] != 0
