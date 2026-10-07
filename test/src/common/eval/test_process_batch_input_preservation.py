"""
Regression tests for GH issue #35.

`process_batch` must not mutate its caller-owned `features`,
`initial_feature_mask`, or `initial_masked_features` arguments, in either
the hard-budget or soft-budget setting.
"""

import torch
from torch.utils.data import TensorDataset

from afabench.evaluation.eval import eval_afa_method, process_batch
from afabench.testing.helpers import (
    get_direct_unmask_fn,
    get_random_afa_predict_fn,
    get_sequential_action_fn,
)


def _run_and_check_inputs_unchanged(selection_budget: float | None) -> None:
    features = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    initial_feature_mask = torch.zeros_like(features, dtype=torch.bool)
    initial_masked_features = torch.zeros_like(features)
    true_label = torch.zeros((2, 3), dtype=torch.float32)

    features_before = features.clone()
    feature_mask_before = initial_feature_mask.clone()
    masked_features_before = initial_masked_features.clone()

    process_batch(
        afa_action_fn=get_sequential_action_fn(),
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=4,
        features=features,
        initial_feature_mask=initial_feature_mask,
        initial_masked_features=initial_masked_features,
        true_label=true_label,
        feature_shape=torch.Size((4,)),
        selection_budget=selection_budget,
    )

    assert torch.equal(features, features_before)
    assert torch.equal(initial_feature_mask, feature_mask_before)
    assert torch.equal(initial_masked_features, masked_features_before)


def test_hard_budget_leaves_caller_owned_inputs_unchanged() -> None:
    _run_and_check_inputs_unchanged(selection_budget=2)


def test_soft_budget_leaves_caller_owned_inputs_unchanged() -> None:
    _run_and_check_inputs_unchanged(selection_budget=None)


class _Dataset(TensorDataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__(features, labels)
        self.feature_shape = torch.Size((features.shape[-1],))

    def get_generation_indices(self) -> torch.Tensor:
        return torch.arange(len(self))


def initialize_half_masked(
    features: torch.Tensor,
    label: torch.Tensor | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> torch.Tensor:
    mask = torch.zeros_like(features, dtype=torch.bool)
    mask[:, : features.shape[-1] // 2] = True
    return mask


def test_eval_afa_method_leaves_dataset_features_unchanged() -> None:
    features = torch.arange(24, dtype=torch.float32).view(6, 4)
    labels = torch.nn.functional.one_hot(
        torch.arange(6) % 3, num_classes=3
    ).float()
    features_before = features.clone()

    eval_afa_method(
        afa_action_fn=get_sequential_action_fn(),
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=4,
        afa_initialize_fn=initialize_half_masked,
        dataset=_Dataset(features, labels),
        builtin_afa_predict_fn=get_random_afa_predict_fn(n_classes=3),
        batch_size=2,
    )

    assert torch.equal(features, features_before)
