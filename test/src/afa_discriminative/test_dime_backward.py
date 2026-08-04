from copy import deepcopy

import torch
from torch import nn


def _dime_like_step(
    template: nn.Module,
    inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    accumulated: bool,
) -> nn.Module:
    model = deepcopy(template)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    losses = [
        nn.functional.mse_loss(model(features), target)
        for features, target in zip(inputs, targets, strict=True)
    ]
    if accumulated:
        torch.stack(losses).mean().backward()
    else:
        for loss in losses:
            (loss / len(losses)).backward()
    optimizer.step()
    return model


def test_one_backward_preserves_dime_optimizer_update() -> None:
    """One traversal must equal DIME's former per-acquisition traversals."""
    torch.manual_seed(7)
    template = nn.Sequential(
        nn.Linear(6, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    inputs = [torch.randn(5, 6) for _ in range(4)]
    targets = [torch.randn(5, 3) for _ in range(4)]

    legacy = _dime_like_step(template, inputs, targets, accumulated=False)
    optimized = _dime_like_step(template, inputs, targets, accumulated=True)

    for legacy_parameter, optimized_parameter in zip(
        legacy.parameters(), optimized.parameters(), strict=True
    ):
        torch.testing.assert_close(
            optimized_parameter,
            legacy_parameter,
            rtol=1e-6,
            atol=1e-7,
        )
    assert torch.equal(
        optimized(inputs[0]).argmax(dim=1),
        legacy(inputs[0]).argmax(dim=1),
    )
