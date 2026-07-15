from typing import TYPE_CHECKING, cast

import torch

from afabench.components.methods.oracle.aaco.core import AACOOracle, get_knn

if TYPE_CHECKING:
    from afabench.core.types import AFAClassifier


class _MaskAwareToyClassifier:
    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        feature_shape: torch.Size | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        probabilities = torch.empty((len(x), 2), device=x.device)
        positive = x[:, 0] > 0
        has_second = mask[:, 1].bool()
        probabilities[positive & has_second] = torch.tensor([0.9, 0.1])
        probabilities[positive & ~has_second] = torch.tensor([0.6, 0.4])
        probabilities[~positive & has_second] = torch.tensor([0.1, 0.9])
        probabilities[~positive & ~has_second] = torch.tensor([0.4, 0.6])
        return probabilities


def _fitted_oracle(objective: str) -> AACOOracle:
    features = torch.tensor([[1.0, 1.0], [-1.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_availability = torch.tensor([[True, True], [True, False]])
    oracle = AACOOracle(
        k_neighbors=2,
        acquisition_cost=0.0,
        missingness_objective=objective,
        dr_max_weight=None,
    )
    oracle.set_classifier(
        cast("AFAClassifier", cast("object", _MaskAwareToyClassifier()))
    )
    oracle.fit(features, labels, observed_mask=source_availability)
    return oracle


def test_support_aware_loss_uses_only_each_neighbors_available_features() -> (
    None
):
    oracle = _fitted_oracle("support_aware")

    loss = oracle._expected_candidate_losses(  # noqa: SLF001
        torch.tensor([[True, True]]),
        torch.tensor([0, 1]),
    )

    expected = (-2 * torch.log(torch.tensor(0.9))) + (
        -2 * torch.log(torch.tensor(0.6))
    )
    assert torch.allclose(loss, expected.unsqueeze(0) / 2)


def test_doubly_robust_loss_corrects_supported_neighbors() -> None:
    oracle = _fitted_oracle("doubly_robust")

    loss = oracle._expected_candidate_losses(  # noqa: SLF001
        torch.tensor([[True, True]]),
        torch.tensor([0, 1]),
    )

    expected = -2 * torch.log(torch.tensor(0.9))
    assert torch.allclose(loss, expected.unsqueeze(0))


def test_knn_distance_ignores_features_missing_from_training_rows() -> None:
    training = torch.tensor(
        [
            [0.0, 1000.0],
            [2.0, 0.0],
            [5.0, 1.0],
        ]
    )
    source_availability = torch.tensor(
        [
            [True, False],
            [True, True],
            [True, True],
        ]
    )

    indices = get_knn(
        training,
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0], [1.0]]),
        num_neighbors=1,
        exclude_instance=False,
        train_observed_mask=source_availability,
    )

    assert int(indices.item()) == 0
