from typing import TYPE_CHECKING, cast, override

import torch
from torch import nn

from afabench.components.methods.rl.jafa.models import (
    LitJAFAEmbedderClassifier,
)

if TYPE_CHECKING:
    from afabench.components.methods.rl.jafa.models import (
        JAFAEmbedder,
        JAFAMLPClassifier,
    )


class _IdentityEmbedder(nn.Module):
    @override
    def forward(
        self, masked_features: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        return masked_features * feature_mask


def _model() -> LitJAFAEmbedderClassifier:
    classifier = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(
            torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]])
        )
    return LitJAFAEmbedderClassifier(
        embedder=cast("JAFAEmbedder", cast("object", _IdentityEmbedder())),
        classifier=cast("JAFAMLPClassifier", cast("object", classifier)),
        class_probabilities=torch.tensor([0.5, 0.5]),
        min_masking_probability=0.0,
        max_masking_probability=0.0,
        lr=1e-3,
    )


def test_unavailable_values_cannot_affect_jafa_pretraining_loss() -> None:
    model = _model()
    availability = torch.tensor([[True, False, True], [False, True, True]])
    original = torch.tensor([[1.0, 20.0, 2.0], [30.0, 4.0, 5.0]])
    perturbed = original.clone()
    perturbed[~availability] = torch.tensor([-900.0, 700.0])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    masked_original, observed_original = model._masked_training_input(  # noqa: SLF001
        original, 0.0, availability
    )
    masked_perturbed, observed_perturbed = model._masked_training_input(  # noqa: SLF001
        perturbed, 0.0, availability
    )

    assert torch.equal(observed_original, availability)
    assert torch.equal(observed_perturbed, availability)
    assert torch.equal(masked_original, masked_perturbed)
    original_loss, _ = model._get_loss_and_acc(  # noqa: SLF001
        masked_original, observed_original, labels
    )
    perturbed_loss, _ = model._get_loss_and_acc(  # noqa: SLF001
        masked_perturbed, observed_perturbed, labels
    )
    assert torch.equal(original_loss, perturbed_loss)
