import pytest
import torch

from afabench.components.methods.rl.ol.config import OLPQModuleConfig
from afabench.components.methods.rl.ol.models import OLPQModule


def _module() -> OLPQModule:
    torch.manual_seed(0)
    return OLPQModule(
        n_features=6,
        n_classes=2,
        n_actions=7,
        cfg=OLPQModuleConfig(
            n_hiddens=[16, 8], p_dropout=0.5, use_feature_mask=False
        ),
    )


def test_eval_mode_predictions_and_qvalues_are_deterministic() -> None:
    """`.eval()` must actually disable dropout on both heads."""
    m = _module().eval()
    x = torch.randn(4, 6)
    with torch.no_grad():
        assert torch.equal(m.forward_q_only(x), m.forward_q_only(x))
        assert torch.equal(m.forward(x)[0], m.forward(x)[0])


def test_train_mode_still_applies_dropout() -> None:
    m = _module().train()
    x = torch.randn(4, 6)
    with torch.no_grad():
        assert not torch.equal(m.forward(x)[0], m.forward(x)[0])


def test_confidence_keeps_mc_dropout_in_eval_mode() -> None:
    """
    The OL trainer computes rewards with the model in eval mode.

    If dropout were disabled there, every MC sample would be identical and the
    confidence reward would carry no information, so `confidence` opts in.
    """
    m = _module().eval()
    x = torch.randn(4, 6)
    with torch.no_grad():
        # Averaging over many stochastic samples must not reproduce the
        # deterministic single-sample forward pass.
        mc = m.confidence(x, mcdrop_samples=64)
        deterministic = m.forward(x)[0].softmax(dim=-1)
    assert not torch.allclose(mc, deterministic, atol=1e-6)
    assert torch.allclose(mc.sum(dim=-1), torch.ones(len(x)), atol=1e-5)


def _availability_sensitive_module(
    *,
    use_action_availability: bool,
) -> OLPQModule:
    module = OLPQModule(
        n_features=2,
        n_classes=2,
        n_actions=3,
        cfg=OLPQModuleConfig(
            n_hiddens=[4],
            p_dropout=0.0,
            use_feature_mask=True,
            use_action_availability=use_action_availability,
        ),
    ).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        if use_action_availability:
            module.layers_q[0].weight[0, 4] = 1.0
            module.layers_q[-1].weight[2, 0] = 1.0
    return module


def test_full_state_qvalues_condition_on_action_availability() -> None:
    module = _availability_sensitive_module(use_action_availability=True)
    values = torch.zeros((1, 2))
    acquired = torch.zeros((1, 2), dtype=torch.bool)

    all_available = module.forward_q_only(
        values,
        acquired,
        torch.tensor([[True, True]]),
    )
    first_unavailable = module.forward_q_only(
        values,
        acquired,
        torch.tensor([[False, True]]),
    )

    assert all_available[0, 2] == 1.0
    assert first_unavailable[0, 2] == 0.0


def test_with_mask_qvalues_do_not_condition_on_action_availability() -> None:
    module = _availability_sensitive_module(use_action_availability=False)
    values = torch.zeros((1, 2))
    acquired = torch.zeros((1, 2), dtype=torch.bool)

    all_available = module.forward_q_only(
        values,
        acquired,
        torch.tensor([[True, True]]),
    )
    first_unavailable = module.forward_q_only(
        values,
        acquired,
        torch.tensor([[False, True]]),
    )

    assert torch.equal(all_available, first_unavailable)


def test_full_state_requires_well_shaped_action_availability() -> None:
    module = _availability_sensitive_module(use_action_availability=True)
    values = torch.zeros((1, 2))
    acquired = torch.zeros((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="was not provided"):
        module.forward_q_only(values, acquired)
    with pytest.raises(ValueError, match="must have shape"):
        module.forward_q_only(
            values,
            acquired,
            torch.ones((1, 3), dtype=torch.bool),
        )

    # Supervised pretraining uses only the classifier head, so the combined
    # forward keeps an all-actions-available default for its unused Q output.
    logits, qvalues = module(values, acquired)
    assert logits.shape == (1, 2)
    assert qvalues.shape == (1, 3)
