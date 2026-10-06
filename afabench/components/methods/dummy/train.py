"""Training logic for the dummy methods, called by their scripts."""

import logging

import torch

from afabench.components.methods.dummy.config import (
    RandomDummyTrainConfig,
    SequentialDummyTrainConfig,
)
from afabench.components.methods.dummy.without_classifier import (
    RandomWithoutClassifierAFAMethod,
    SequentialWithoutClassifierAFAMethod,
)
from afabench.core.types import AFAMethod
from afabench.evaluation.eval import eval_afa_method
from afabench.training.inputs import TrainingInputs
from afabench.training.smoke_test import eval_settings

log = logging.getLogger(__name__)


def train_random_dummy(
    contract: RandomDummyTrainConfig, inputs: TrainingInputs
) -> RandomWithoutClassifierAFAMethod:
    train_dataset = inputs.train_dataset()
    assert len(train_dataset.label_shape) == 1, "Only 1D labels supported"

    afa_method = RandomWithoutClassifierAFAMethod(
        device=torch.device("cpu"),
        n_classes=train_dataset.label_shape.numel(),
        prob_select_0=0.0
        if contract.soft_budget_param is None
        else contract.soft_budget_param,
    )
    _check_dummy_method_works(contract, inputs, afa_method)
    return afa_method


def train_sequential_dummy(
    contract: SequentialDummyTrainConfig, inputs: TrainingInputs
) -> SequentialWithoutClassifierAFAMethod:
    train_dataset = inputs.train_dataset()
    assert len(train_dataset.label_shape) == 1, "Only 1D labels supported"

    afa_method = SequentialWithoutClassifierAFAMethod(
        device=torch.device("cpu"),
        n_classes=train_dataset.label_shape[-1],
        prob_select_0=0.0
        if contract.soft_budget_param is None
        else contract.soft_budget_param,
    )
    _check_dummy_method_works(contract, inputs, afa_method)
    return afa_method


def _check_dummy_method_works(
    contract: RandomDummyTrainConfig | SequentialDummyTrainConfig,
    inputs: TrainingInputs,
    afa_method: AFAMethod,
) -> None:
    """Check the method works end to end; dummy methods have no real training."""
    train_dataset = inputs.train_dataset()
    initializer = inputs.initializer()
    unmasker = inputs.unmasker()
    only_n_samples, batch_size = eval_settings(
        smoke_test=contract.smoke_test,
        default_n_samples=100,
        default_batch_size=10,
    )
    eval_afa_method(
        afa_action_fn=afa_method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=unmasker.get_n_selections(
            train_dataset.feature_shape
        ),
        afa_initialize_fn=initializer.initialize,
        dataset=train_dataset,
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=afa_method.predict,
        only_n_samples=only_n_samples,
        batch_size=batch_size,
    )
