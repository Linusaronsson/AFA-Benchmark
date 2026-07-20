"""
Batching the instance dimension must not couple instances together.

This is what `eval_batch_size` being a throughput knob rests on: scoring B
instances together has to give each of them the answer it would have got alone.
"""

from typing import TYPE_CHECKING, cast

import pytest
import torch

from afabench.components.methods.oracle.aaco.core import (
    AACOOracle,
    get_knn_batched,
)

if TYPE_CHECKING:
    from afabench.core.types import AFAClassifier

N_TRAIN, N_FEATURES, N_INSTANCES = 60, 8, 12


class _SmoothToyClassifier:
    """A cheap classifier whose output depends on both values and mask."""

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        feature_shape: torch.Size | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        score = (x * mask).sum(dim=-1, keepdim=True) + 0.25 * mask.sum(
            dim=-1, keepdim=True
        )
        return torch.cat([score, -score], dim=-1).softmax(dim=-1)


def _fitted_oracle(
    objective: str, *, restricted: bool, acquisition_cost: float
) -> tuple[AACOOracle, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    features = torch.randn(N_TRAIN, N_FEATURES, generator=g)
    labels = torch.zeros(N_TRAIN, 2)
    labels[torch.arange(N_TRAIN), (features[:, 0] > 0).long()] = 1.0
    availability = (
        (torch.rand(N_TRAIN, N_FEATURES, generator=g) > 0.3)
        if restricted
        else None
    )
    oracle = AACOOracle(
        k_neighbors=3,
        acquisition_cost=acquisition_cost,
        missingness_objective=objective,
        mask_seed=0,
    )
    oracle.set_classifier(
        cast("AFAClassifier", cast("object", _SmoothToyClassifier()))
    )
    oracle.fit(features, labels, observed_mask=availability)
    return oracle, features


def _queries() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    x = torch.randn(N_INSTANCES, N_FEATURES, generator=g)
    # A spread of observation counts, including the empty-support edge case.
    mask = torch.rand(N_INSTANCES, N_FEATURES, generator=g) > 0.5
    mask[0] = False
    mask[1] = True
    return x, mask


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("force_acquisition", [False, True])
@pytest.mark.parametrize("objective", ["support_aware", "doubly_robust"])
def test_batched_matches_one_instance_at_a_time(
    *, restricted: bool, force_acquisition: bool, objective: str
) -> None:
    oracle, _ = _fitted_oracle(
        objective, restricted=restricted, acquisition_cost=0.05
    )
    x, mask = _queries()
    instance_idx = torch.arange(N_INSTANCES)

    batched = oracle.select_next_features_batched(
        x,
        mask,
        instance_idx=instance_idx,
        force_acquisition=force_acquisition,
        exclude_instance=False,
    )
    one_at_a_time = [
        oracle.select_next_features_batched(
            x[i : i + 1],
            mask[i : i + 1],
            instance_idx=instance_idx[i : i + 1],
            force_acquisition=force_acquisition,
            exclude_instance=False,
        )[0]
        for i in range(N_INSTANCES)
    ]
    assert batched == one_at_a_time


def test_soft_budget_can_still_choose_to_stop() -> None:
    """A high acquisition cost must make stopping win, not just be reachable."""
    oracle, _ = _fitted_oracle(
        "support_aware", restricted=False, acquisition_cost=1e6
    )
    x, mask = _queries()

    chosen = oracle.select_next_features_batched(
        x, mask, force_acquisition=False, exclude_instance=False
    )
    assert all(c is None for c in chosen)


def test_forced_acquisition_returns_a_feature_unless_nothing_is_left() -> None:
    """
    Under a hard budget an instance may not stop while it still has a choice.

    Instance 1 of the fixture is fully observed, and there None is the only
    honest answer; the evaluator, not the oracle, owns that case.
    """
    oracle, _ = _fitted_oracle(
        "support_aware", restricted=False, acquisition_cost=1e6
    )
    x, mask = _queries()

    chosen = oracle.select_next_features_batched(
        x, mask, force_acquisition=True, exclude_instance=False
    )
    exhausted = mask.all(dim=1)
    assert exhausted.any(), "fixture no longer covers the exhausted case"
    for i, c in enumerate(chosen):
        if exhausted[i]:
            assert c is None
        else:
            assert c is not None
            # And never a feature that is already observed.
            assert not mask[i, c]


N_SELECTIONS = 1 + (N_FEATURES - 2)


def _selection_to_feature() -> torch.Tensor:
    """Build a cube_nm-shaped table: selection 0 groups two features, rest 1:1."""
    table = torch.zeros(N_SELECTIONS, N_FEATURES, dtype=torch.bool)
    table[0, :2] = True
    for s in range(1, N_SELECTIONS):
        table[s, s + 1] = True
    return table


def _selection_queries() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(13)
    x = torch.randn(N_INSTANCES, N_FEATURES, generator=g)
    taken = torch.rand(N_INSTANCES, N_SELECTIONS, generator=g) > 0.6
    taken[0] = False  # nothing taken yet
    taken[1] = True  # nothing left to take
    observed = (taken.float() @ _selection_to_feature().float()) > 0
    return x, observed, taken


def _reference_selection(
    oracle: AACOOracle,
    observed: torch.Tensor,
    taken: torch.Tensor,
    idx_nn: torch.Tensor,
    *,
    force_acquisition: bool,
) -> list[int | None]:
    """
    Run the greedy one-step search over selections, written out literally.

    For each instance, score adding each still-available selection on its own,
    plus the option to stop, and take the cheapest. Deliberately a slow Python
    loop scoring one candidate at a time: it is the specification the
    rectangular batched path has to match, not a second copy of it.

    Neighbours are passed in rather than recomputed, so a disagreement here is
    about candidate scoring and never about which neighbours the KNN picked.
    Those have their own tests in `test_aaco_knn_batching.py`.
    """
    table = _selection_to_feature()

    def loss(mask: torch.Tensor, b: int) -> float:
        return float(
            oracle._expected_candidate_losses(  # noqa: SLF001
                mask.reshape(1, 1, -1),
                idx_nn[b : b + 1],
            )[0, 0]
        )

    out: list[int | None] = []
    for b in range(observed.shape[0]):
        best_cost, best = torch.inf, None
        if not force_acquisition:
            best_cost = loss(observed[b], b)
        for s in range(table.shape[0]):
            if taken[b, s]:
                continue
            cost = loss(observed[b] | table[s], b) + (
                oracle.acquisition_cost
                * float((table[s] & ~observed[b]).sum())
            )
            if cost < best_cost:
                best_cost, best = cost, s
        out.append(best)
    return out


@pytest.mark.parametrize("force_acquisition", [False, True])
def test_selection_path_matches_the_greedy_one_step_spec(
    *, force_acquisition: bool
) -> None:
    """Pin the semantics now that the serial implementation is gone."""
    oracle, _ = _fitted_oracle(
        "support_aware", restricted=False, acquisition_cost=0.05
    )
    x, observed, taken = _selection_queries()
    assert oracle.X_train is not None

    idx_nn = get_knn_batched(
        oracle.X_train,
        x,
        observed.float().T,
        oracle.k_neighbors,
        exclude_instance=False,
    ).T

    batched = oracle.select_next_selections_batched(
        x,
        observed,
        taken,
        _selection_to_feature(),
        instance_idx=torch.arange(N_INSTANCES),
        force_acquisition=force_acquisition,
        exclude_instance=False,
    )
    want = _reference_selection(
        oracle, observed, taken, idx_nn, force_acquisition=force_acquisition
    )
    assert batched == want


@pytest.mark.parametrize("force_acquisition", [False, True])
def test_selection_batching_matches_one_instance_at_a_time(
    *, force_acquisition: bool
) -> None:
    oracle, _ = _fitted_oracle(
        "doubly_robust", restricted=True, acquisition_cost=0.05
    )
    x, observed, taken = _selection_queries()
    table = _selection_to_feature()
    instance_idx = torch.arange(N_INSTANCES)

    batched = oracle.select_next_selections_batched(
        x,
        observed,
        taken,
        table,
        instance_idx=instance_idx,
        force_acquisition=force_acquisition,
        exclude_instance=False,
    )
    one_at_a_time = [
        oracle.select_next_selections_batched(
            x[i : i + 1],
            observed[i : i + 1],
            taken[i : i + 1],
            table,
            instance_idx=instance_idx[i : i + 1],
            force_acquisition=force_acquisition,
            exclude_instance=False,
        )[0]
        for i in range(N_INSTANCES)
    ]
    assert batched == one_at_a_time


def test_a_taken_selection_is_never_returned() -> None:
    """
    Scoring every selection, taken or not, is what keeps the pass rectangular.

    The taken ones are excluded by cost rather than by being left out, so this
    is the assertion that the masking actually holds.
    """
    oracle, _ = _fitted_oracle(
        "support_aware", restricted=False, acquisition_cost=0.0
    )
    x, observed, taken = _selection_queries()

    chosen = oracle.select_next_selections_batched(
        x,
        observed,
        taken,
        _selection_to_feature(),
        force_acquisition=True,
        exclude_instance=False,
    )
    exhausted = taken.all(dim=1)
    assert exhausted.any(), "fixture no longer covers the exhausted case"
    for i, c in enumerate(chosen):
        if exhausted[i]:
            assert c is None
        else:
            assert c is not None
            assert not taken[i, c]


def test_tiebreak_path_is_exercised_and_batches() -> None:
    """
    Guard the equivalence test above against silently skipping the tie-break.

    A zero acquisition cost makes large candidate masks attractive, so the
    winning mask routinely adds several features at once and the single-feature
    ordering pass has to run. Without this the parametrized test could pass
    while never touching that branch.
    """
    oracle, _ = _fitted_oracle(
        "support_aware", restricted=False, acquisition_cost=0.0
    )
    x, mask = _queries()

    calls = {"n": 0}
    original = AACOOracle._expected_candidate_losses  # noqa: SLF001

    def counting(self, masks, idx_nn):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return original(self, masks, idx_nn)

    AACOOracle._expected_candidate_losses = counting  # noqa: SLF001
    try:
        batched = oracle.select_next_features_batched(
            x, mask, force_acquisition=True, exclude_instance=False
        )
        assert calls["n"] == 2, "expected the tie-break pass to run"
        one_at_a_time = [
            oracle.select_next_features_batched(
                x[i : i + 1],
                mask[i : i + 1],
                force_acquisition=True,
                exclude_instance=False,
            )[0]
            for i in range(N_INSTANCES)
        ]
    finally:
        AACOOracle._expected_candidate_losses = original  # noqa: SLF001

    assert batched == one_at_a_time
