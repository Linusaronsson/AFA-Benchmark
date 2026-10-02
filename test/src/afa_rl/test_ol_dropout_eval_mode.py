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
