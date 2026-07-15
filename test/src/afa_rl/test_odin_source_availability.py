from typing import cast

import pytest
import torch
from torch import nn

from afabench.components.methods.rl.odin.models import (
    ODINPretrainingModel,
    PartialVAE,
)


def _model() -> ODINPretrainingModel:
    return ODINPretrainingModel(
        partial_vae=cast("PartialVAE", nn.Identity()),
        classifier=nn.Identity(),
        class_probabilities=torch.tensor([1.0]),
        min_masking_probability=0.1,
        max_masking_probability=0.9,
        lr=1e-3,
        start_kl_scaling_factor=0.1,
        end_kl_scaling_factor=0.1,
        n_annealing_epochs=1,
        classifier_loss_scaling_factor=1.0,
    )


def test_reconstruction_loss_excludes_unavailable_targets() -> None:
    model = _model()
    estimated = torch.zeros((1, 2))
    features = torch.tensor([[1.0, 100.0]])
    zeros = torch.zeros((1, 1))

    _, restricted_loss, _ = model.partial_vae_loss_function(
        estimated,
        features,
        zeros,
        zeros,
        reconstruction_availability=torch.tensor([[True, False]]),
    )
    _, complete_loss, _ = model.partial_vae_loss_function(
        estimated,
        features,
        zeros,
        zeros,
    )

    assert float(restricted_loss) == pytest.approx(1.0)
    assert float(complete_loss) == pytest.approx(10001.0)
