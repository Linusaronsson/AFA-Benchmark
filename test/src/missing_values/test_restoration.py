from typing import TYPE_CHECKING, cast

import torch

from afabench.datasets.datasets import CubeNMDataset
from afabench.datasets.training_views import restricted_training_view
from afabench.missing_values.restoration import restore_view_with_pvae

if TYPE_CHECKING:
    from afabench.components.methods.rl.odin.models import ODINPretrainingModel


class _FakePVAE:
    device = torch.device("cpu")

    def masked_reconstruction(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,  # noqa: ARG002
        n_classes: int,  # noqa: ARG002
        label: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fill = 7.0 if label is None else 9.0
        return torch.zeros((len(masked_features), 1)), torch.full_like(
            masked_features,
            fill,
        )


def test_pvae_restoration_uses_requested_label_mode() -> None:
    dataset = CubeNMDataset(n_samples=4, seed=1, n_contexts=2)
    availability = torch.ones_like(dataset.features, dtype=torch.bool)
    availability[:, ::2] = False
    restricted = restricted_training_view(dataset, availability)
    model = cast("ODINPretrainingModel", cast("object", _FakePVAE()))

    label_free = restore_view_with_pvae(
        restricted,
        model,
        strategy="pvae_label_free",
        batch_size=2,
    )
    conditioned = restore_view_with_pvae(
        restricted,
        model,
        strategy="pvae_label_conditioned",
        batch_size=3,
    )

    assert (label_free.features[~availability] == 7.0).all()
    assert (conditioned.features[~availability] == 9.0).all()
    assert torch.equal(
        conditioned.features[availability],
        dataset.features[availability],
    )
