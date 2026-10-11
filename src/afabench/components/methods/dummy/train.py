"""Training logic for the dummy methods, called by their scripts."""

import logging
from collections.abc import Callable

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
from afabench.fit.inputs import FitInputs
from afabench.fit.smoke_test import eval_settings

log = logging.getLogger(__name__)


def train_random_dummy(
    contract: RandomDummyTrainConfig, inputs: FitInputs
) -> RandomWithoutClassifierAFAMethod:
    return _train_dummy_method(
        RandomWithoutClassifierAFAMethod, contract, inputs
    )


def train_sequential_dummy(
    contract: SequentialDummyTrainConfig, inputs: FitInputs
) -> SequentialWithoutClassifierAFAMethod:
    return _train_dummy_method(
        SequentialWithoutClassifierAFAMethod, contract, inputs
    )


def _train_dummy_method[M: AFAMethod](
    method_class: Callable[..., M],
    contract: RandomDummyTrainConfig | SequentialDummyTrainConfig,
    inputs: FitInputs,
) -> M:
    train_dataset = inputs.train_dataset()
    if len(train_dataset.label_shape) != 1:
        msg = (
            "Only 1D labels are supported, got "
            f"label_shape={train_dataset.label_shape}"
        )
        raise ValueError(msg)

    afa_method = method_class(
        device=torch.device("cpu"),
        n_classes=train_dataset.label_shape.numel(),
        prob_select_0=0.0
        if contract.soft_budget_param is None
        else contract.soft_budget_param,
    )
    _check_dummy_method_works(contract, inputs, afa_method)
    return afa_method


def _check_dummy_method_works(
    contract: RandomDummyTrainConfig | SequentialDummyTrainConfig,
    inputs: FitInputs,
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
