"""DIME training, on tabular and image datasets alike."""

import logging
from collections.abc import Callable

import torch
from torch import nn

from afabench.components.methods.discriminative.common.modality import (
    prepare_modality,
)
from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
)
from afabench.components.methods.discriminative.common.trainer import (
    NetworkPairTrainer,
    TrainerSettings,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
    DIMETrainingObjective,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMETrainingConfig,
)
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)

# The upstream CMIEstimator default, used when the config sets no min_lr;
# the tabular configs set none
UPSTREAM_MIN_LR = 1e-6


def train(
    cfg: DIMETrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> DIMEAFAMethod:
    log.debug(cfg)
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    assert cfg.pretrained_model_bundle_path is not None, (
        "pretrained_model_bundle_path must be configured"
    )
    torch.set_float32_matmul_precision("medium")
    classifier_bundle = inputs.pretrained_model(GreedyAFAClassifier)
    predictor = classifier_bundle.predictor.to(torch.device(cfg.device))
    setup = prepare_modality(
        cfg, inputs, classifier_arch=classifier_bundle.architecture
    )
    value_network = setup.policy_network(share_with=predictor)
    setup.check_policy_output(value_network)
    trainer = NetworkPairTrainer(
        policy_network=value_network,
        predictor=predictor,
        mask_layer=setup.mask_layer,
        initializer=inputs.initializer(),
        unmasker=inputs.unmasker(),
        objective=DIMETrainingObjective(
            eps=cfg.eps, eps_decay=cfg.eps_decay, eps_steps=cfg.eps_steps
        ),
    )
    trainer.fit(
        setup.train_loader,
        setup.val_loader,
        TrainerSettings(
            lr=cfg.lr,
            nepochs=cfg.nepochs,
            selections_per_episode=cfg.hard_budget,
            loss_fn=nn.CrossEntropyLoss(
                reduction="none", weight=setup.class_weights
            ),
            min_lr=setup.min_lr(cfg.min_lr, upstream_default=UPSTREAM_MIN_LR),
            patience=cfg.patience,
        ),
        feature_costs=setup.feature_costs,
        metric_logger=metric_logger,
    )
    return DIMEAFAMethod(
        value_network.cpu(),
        predictor.cpu(),
        setup.network_pair_spec(),
        torch.device("cpu"),
        selection_costs=setup.selection_costs,
    )
