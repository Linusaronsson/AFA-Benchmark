from copy import deepcopy

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import register_optimizer_step_pre_hook
from torch.utils.data import DataLoader, TensorDataset

from afabench.components.initializers.zero_initializer import ZeroInitializer
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
    get_entropy,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    CMIEstimator,
)
from afabench.components.unmaskers.direct_unmasker import DirectUnmasker

N_SAMPLES = 6
N_FEATURES = 4
N_CLASSES = 3
MAX_FEATURES = 3


def _per_acquisition_backward_grads(
    value_network: nn.Module,
    predictor: nn.Module,
    features: torch.Tensor,
    label: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute DIME gradients with one backward pass per acquisition."""
    mask_layer = MaskLayer(append=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    feature_mask = torch.zeros_like(features)

    pred_without_next_feature = predictor(mask_layer(features, feature_mask))
    loss_without_next_feature = loss_fn(pred_without_next_feature, label)
    (loss_without_next_feature.mean() / (MAX_FEATURES + 1)).backward()
    pred_without_next_feature = pred_without_next_feature.detach()
    loss_without_next_feature = loss_without_next_feature.detach()

    for _ in range(MAX_FEATURES):
        masked_features = mask_layer(features, feature_mask)
        entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
        pred_cmi = value_network(masked_features).sigmoid() * entropy
        # Unit feature costs and eps=0 make every acquisition greedy.
        actions = pred_cmi.argmax(dim=1)
        feature_mask = feature_mask.clone()
        feature_mask[torch.arange(N_SAMPLES), actions] = 1.0

        pred_with_next_feature = predictor(mask_layer(features, feature_mask))
        loss_with_next_feature = loss_fn(pred_with_next_feature, label)
        delta = loss_without_next_feature - loss_with_next_feature.detach()
        value_network_loss = nn.functional.mse_loss(
            pred_cmi[torch.arange(N_SAMPLES), actions], delta
        )
        total_loss = value_network_loss + loss_with_next_feature.mean()
        (total_loss / (MAX_FEATURES + 1)).backward()

        loss_without_next_feature = loss_with_next_feature.detach()
        pred_without_next_feature = pred_with_next_feature.detach()

    parameters = [*value_network.parameters(), *predictor.parameters()]
    return [
        parameter.grad.clone()
        for parameter in parameters
        if parameter.grad is not None
    ]


def test_dime_single_backward_matches_per_acquisition_backward() -> None:
    """One backward pass over all acquisitions yields the same gradients."""
    torch.manual_seed(7)
    features = torch.randn(N_SAMPLES, N_FEATURES)
    label = torch.randint(N_CLASSES, (N_SAMPLES,))
    value_network = nn.Sequential(
        nn.Linear(2 * N_FEATURES, 16),
        nn.ReLU(),
        nn.Linear(16, N_FEATURES),
    )
    predictor = nn.Sequential(
        nn.Linear(2 * N_FEATURES, 16),
        nn.ReLU(),
        nn.Linear(16, N_CLASSES),
    )
    expected_grads = _per_acquisition_backward_grads(
        deepcopy(value_network), deepcopy(predictor), features, label
    )

    captured_grads: list[list[torch.Tensor]] = []

    def capture_grads(
        optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        captured_grads.append(
            [
                parameter.grad.clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
        )

    loader = DataLoader(
        TensorDataset(features, label), batch_size=N_SAMPLES, shuffle=False
    )
    estimator = CMIEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=MaskLayer(append=True),
        initializer=ZeroInitializer(),
        unmasker=DirectUnmasker(),
    )
    handle = register_optimizer_step_pre_hook(capture_grads)
    try:
        estimator.fit(
            loader,
            loader,
            lr=1e-3,
            nepochs=1,
            max_features=MAX_FEATURES,
            eps=0.0,
            loss_fn=nn.CrossEntropyLoss(reduction="none"),
            val_loss_fn=None,
            val_loss_mode=None,
            verbose=False,
        )
    finally:
        handle.remove()

    assert len(captured_grads) == 1
    (actual_grads,) = captured_grads
    n_parameters = len([*value_network.parameters(), *predictor.parameters()])
    assert len(actual_grads) == len(expected_grads) == n_parameters
    assert all(grad.abs().sum() > 0 for grad in expected_grads)
    for actual_grad, expected_grad in zip(
        actual_grads, expected_grads, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
