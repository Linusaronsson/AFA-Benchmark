import torch

from afabench.common.cube_nm_ar_oracle_method import CubeNMAROracleMethod
from afabench.common.unmaskers.cube_nm_ar_unmasker import CubeNMARUnmasker
from afabench.eval.eval import process_batch

N_CONTEXTS = 5
N_HINTS = 5
N_ADMIN = 2
BLOCK_SIZE = 4
N_FEATURES = N_CONTEXTS + N_HINTS + N_ADMIN + N_CONTEXTS * BLOCK_SIZE + 1
N_SELECTIONS = 1 + N_CONTEXTS * BLOCK_SIZE + 1
SELECTABLE_START = N_CONTEXTS + N_HINTS + N_ADMIN
RESCUE_ACTION = 2 + N_CONTEXTS * BLOCK_SIZE


def _make_method(max_cost: float | None = None) -> CubeNMAROracleMethod:
    return CubeNMAROracleMethod(
        n_contexts=N_CONTEXTS,
        n_safe_contexts=2,
        n_hint_features=N_HINTS,
        n_admin_features=N_ADMIN,
        block_size=BLOCK_SIZE,
        n_classes=8,
        context_action_cost=1.0,
        selectable_feature_costs=torch.tensor([1.0] * 20 + [4.0]),
        device=torch.device("cpu"),
        max_cost=max_cost,
    )


def _blank_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_features = torch.zeros((1, N_FEATURES), dtype=torch.float32)
    feature_mask = torch.zeros((1, N_FEATURES), dtype=torch.bool)
    selection_mask = torch.zeros((1, N_SELECTIONS), dtype=torch.bool)
    return masked_features, feature_mask, selection_mask


def _observe_action(
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    selection_mask: torch.Tensor,
    *,
    context_idx: int,
    action: int,
) -> None:
    if action == 1:
        feature_mask[0, :N_CONTEXTS] = True
        masked_features[0, :N_CONTEXTS] = 0.0
        masked_features[0, context_idx] = 1.0
    else:
        feature_idx = SELECTABLE_START + (action - 2)
        feature_mask[0, feature_idx] = True
        masked_features[0, feature_idx] = 1.0
    selection_mask[0, action - 1] = True


def test_cube_nm_ar_oracle_follows_risky_context_plan_from_cold_start() -> (
    None
):
    method = _make_method()
    masked_features, feature_mask, selection_mask = _blank_state()

    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 1
    )

    _observe_action(
        masked_features,
        feature_mask,
        selection_mask,
        context_idx=3,
        action=1,
    )
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 14
    )

    _observe_action(
        masked_features,
        feature_mask,
        selection_mask,
        context_idx=3,
        action=14,
    )
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 15
    )

    _observe_action(
        masked_features,
        feature_mask,
        selection_mask,
        context_idx=3,
        action=15,
    )
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 22
    )

    _observe_action(
        masked_features,
        feature_mask,
        selection_mask,
        context_idx=3,
        action=22,
    )
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 0
    )


def test_cube_nm_ar_oracle_uses_unique_hint_to_skip_context() -> None:
    method = _make_method()
    masked_features, feature_mask, selection_mask = _blank_state()

    hint_idx = N_CONTEXTS + 1
    feature_mask[0, hint_idx] = True
    masked_features[0, hint_idx] = 1.0

    # Safe context 1 starts at action 6.
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 6
    )


def test_cube_nm_ar_oracle_respects_cost_ceiling() -> None:
    method = _make_method(max_cost=0.5)
    masked_features, feature_mask, selection_mask = _blank_state()
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 0
    )

    method = _make_method(max_cost=1.0)
    masked_features, feature_mask, selection_mask = _blank_state()
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 1
    )
    _observe_action(
        masked_features,
        feature_mask,
        selection_mask,
        context_idx=4,
        action=1,
    )
    assert (
        method.act(masked_features, feature_mask, selection_mask).item() == 0
    )


def test_cube_nm_ar_oracle_matches_budget_6_vs_7_landmark() -> None:
    method = _make_method()
    unmasker = CubeNMARUnmasker(
        n_contexts=N_CONTEXTS,
        n_hint_features=N_HINTS,
        n_admin_features=N_ADMIN,
    )
    features = torch.zeros((1, N_FEATURES), dtype=torch.float32)
    features[0, 4] = 1.0
    true_label = torch.nn.functional.one_hot(
        torch.tensor([0]), num_classes=8
    ).float()
    initial_feature_mask = torch.zeros_like(features, dtype=torch.bool)
    initial_masked_features = torch.zeros_like(features)

    budget6_df = process_batch(
        afa_action_fn=method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=N_SELECTIONS,
        features=features,
        initial_feature_mask=initial_feature_mask,
        initial_masked_features=initial_masked_features,
        true_label=true_label,
        selection_budget=6,
        selection_costs=[1.0] + [1.0] * 20 + [4.0],
    )
    assert budget6_df["action_performed"].tolist() == [1, 18, 19, 0]
    assert budget6_df["forced_stop"].tolist() == [False, False, False, True]

    budget7_df = process_batch(
        afa_action_fn=method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=N_SELECTIONS,
        features=features,
        initial_feature_mask=initial_feature_mask,
        initial_masked_features=initial_masked_features,
        true_label=true_label,
        selection_budget=7,
        selection_costs=[1.0] + [1.0] * 20 + [4.0],
    )
    assert budget7_df["action_performed"].tolist() == [1, 18, 19, 22, 0]
    assert budget7_df["forced_stop"].tolist() == [
        False,
        False,
        False,
        False,
        False,
    ]
