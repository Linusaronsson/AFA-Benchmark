from collections.abc import Sequence

import pandas as pd
import torch

from afabench.common.custom_types import (
    AFAAction,
    AFAActionFn,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


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
