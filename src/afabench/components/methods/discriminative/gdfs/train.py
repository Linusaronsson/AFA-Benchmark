"""GDFS training, on tabular and image datasets alike."""

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
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
    GDFSTrainingObjective,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSTrainingConfig,
)
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)

# The upstream GreedyDynamicSelection.fit default, used when the config
# sets no min_lr; the tabular configs set none
UPSTREAM_MIN_LR = 1e-5


def train(
    cfg: GDFSTrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> GDFSAFAMethod:
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
    selector = setup.policy_network()
    setup.check_policy_output(selector)
    trainer = NetworkPairTrainer(
        policy_network=selector,
        predictor=predictor,
        mask_layer=setup.mask_layer,
        initializer=inputs.initializer(),
        unmasker=inputs.unmasker(),
        objective=GDFSTrainingObjective(temp_steps=1 if cfg.smoke_test else 5),
    )
    trainer.fit(
        setup.train_loader,
        setup.val_loader,
        TrainerSettings(
            lr=cfg.lr,
            nepochs=cfg.nepochs,
            selections_per_episode=cfg.hard_budget,
            loss_fn=nn.CrossEntropyLoss(weight=setup.class_weights),
            min_lr=setup.min_lr(cfg.min_lr, upstream_default=UPSTREAM_MIN_LR),
            patience=cfg.patience,
        ),
        feature_costs=setup.feature_costs,
        metric_logger=metric_logger,
    )
    return GDFSAFAMethod(
        selector.cpu(),
        predictor.cpu(),
        setup.network_pair_spec(),
        torch.device("cpu"),
        selection_costs=setup.selection_costs,
    )
