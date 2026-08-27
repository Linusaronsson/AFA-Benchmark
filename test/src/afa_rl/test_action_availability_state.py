from typing import TYPE_CHECKING, cast, override

import torch
from torch import nn

from afabench.components.methods.rl.jafa.agents import JAFAActionValueModule
from afabench.components.methods.rl.odin.agents import (
    ODINPolicyModule,
    ODINValueModule,
)

if TYPE_CHECKING:
    from afabench.components.methods.rl.jafa.models import JAFAEmbedder


class _IdentityEmbedder(nn.Module):
    @override
    def forward(
        self, masked_features: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        return masked_features * feature_mask


def _only_linear(module: nn.Module) -> nn.Linear:
    layers = [
        layer for layer in module.modules() if isinstance(layer, nn.Linear)
    ]
    assert len(layers) == 1
    return layers[0]


def _jafa(*, use_action_availability: bool) -> JAFAActionValueModule:
    module = JAFAActionValueModule(
        embedder=cast("JAFAEmbedder", cast("object", _IdentityEmbedder())),
        embedding_size=2,
        action_size=3,
        num_cells=(),
        dropout=0.0,
        n_feature_dims=1,
        use_action_availability=use_action_availability,
    ).eval()
    with torch.no_grad():
        linear = _only_linear(module.net)
        linear.weight.zero_()
        linear.bias.zero_()
        if use_action_availability:
            linear.weight[2, 2] = 1.0
    return module


def test_jafa_state_variants_separate_conditioning_from_legality() -> None:
    features = torch.zeros((1, 2))
    acquired = torch.ones((1, 2), dtype=torch.bool)
    all_legal = torch.tensor([[True, True, True]])
    first_blocked = torch.tensor([[True, False, True]])

    mask_free = _jafa(use_action_availability=False)
    all_legal_mask_free = mask_free(features, acquired, all_legal)
    first_blocked_mask_free = mask_free(features, acquired, first_blocked)
    assert torch.equal(
        all_legal_mask_free[:, [0, 2]],
        first_blocked_mask_free[:, [0, 2]],
    )
    assert first_blocked_mask_free[0, 1].isneginf()

    full_state = _jafa(use_action_availability=True)
    all_legal_values = full_state(features, acquired, all_legal)
    first_blocked_values = full_state(features, acquired, first_blocked)
    assert all_legal_values[0, 2] == 1.0
    assert first_blocked_values[0, 2] == 0.0
    assert first_blocked_values[0, 1].isneginf()


def _odin_policy(*, use_action_availability: bool) -> ODINPolicyModule:
    module = ODINPolicyModule(
        latent_size=2,
        n_actions=3,
        num_cells=(),
        dropout=0.0,
        use_action_availability=use_action_availability,
    ).eval()
    with torch.no_grad():
        linear = _only_linear(module.net)
        linear.weight.zero_()
        linear.bias.zero_()
        if use_action_availability:
            linear.weight[2, 2] = 1.0
    return module


def _odin_value(*, use_action_availability: bool) -> ODINValueModule:
    module = ODINValueModule(
        latent_size=2,
        n_actions=3,
        num_cells=(),
        dropout=0.0,
        use_action_availability=use_action_availability,
    ).eval()
    with torch.no_grad():
        linear = _only_linear(module.net)
        linear.weight.zero_()
        linear.bias.zero_()
        if use_action_availability:
            linear.weight[0, 2] = 1.0
    return module


def test_odin_state_variants_condition_both_actor_and_critic() -> None:
    mu = torch.zeros((1, 2))
    all_legal = torch.tensor([[True, True, True]])
    first_blocked = torch.tensor([[True, False, True]])

    mask_free_policy = _odin_policy(use_action_availability=False)
    mask_free_value = _odin_value(use_action_availability=False)
    all_legal_mask_free = mask_free_policy(mu, all_legal)
    first_blocked_mask_free = mask_free_policy(mu, first_blocked)
    assert torch.equal(
        all_legal_mask_free[:, [0, 2]],
        first_blocked_mask_free[:, [0, 2]],
    )
    assert first_blocked_mask_free[0, 1].isneginf()
    assert torch.equal(
        mask_free_value(mu, all_legal),
        mask_free_value(mu, first_blocked),
    )

    full_state_policy = _odin_policy(use_action_availability=True)
    full_state_value = _odin_value(use_action_availability=True)
    all_legal_logits = full_state_policy(mu, all_legal)
    first_blocked_logits = full_state_policy(mu, first_blocked)
    assert all_legal_logits[0, 2] == 1.0
    assert first_blocked_logits[0, 2] == 0.0
    assert first_blocked_logits[0, 1].isneginf()
    assert full_state_value(mu, all_legal).item() == 1.0
    assert full_state_value(mu, first_blocked).item() == 0.0
