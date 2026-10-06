import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
    MaskingPretrainer,
)
from afabench.components.methods.discriminative.common.utils import MaskLayer
from afabench.components.methods.discriminative.dime.config import (
    DIMEPretrainingConfig,
    DIMETabularArchitectureConfig,
)
from afabench.core.utils import get_class_frequencies
from afabench.training.inputs import TrainingInputs

log = logging.getLogger(__name__)


def pretrain_tabular(
    cfg: DIMEPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: TrainingInputs,
) -> GreedyAFAClassifier:
    log.debug(cfg)
    assert isinstance(cfg.architecture, DIMETabularArchitectureConfig)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    val_dataset = inputs.val_dataset()
    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    class_weights = len(train_class_probabilities) / (
        len(train_class_probabilities) * train_class_probabilities
    )
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )
    in_features: int = int(d_in * 2)
    out_features: int = int(d_out)
    hidden_units = cfg.architecture.hidden_units
    activation_name: str = cfg.architecture.activation
    dropout: float = float(cfg.architecture.dropout)
    predictor = MLP(
        in_features=in_features,
        out_features=out_features,
        num_cells=hidden_units,
        activation_class=getattr(nn, activation_name),
        dropout=dropout,
    )
    architecture: dict[str, Any] = {
        "type": "mlp",
        "in_features": in_features,
        "out_features": out_features,
        "hidden_units": hidden_units,
        "activation": activation_name,
        "dropout": dropout,
    }
    mask_layer = MaskLayer(append=True)
    print("Pretraining predictor")
    print("-" * 8)
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
        metric_logger=metric_logger,
        metric_prefix="dime_pretrain",
    )
    bundle_obj = GreedyAFAClassifier(
        predictor=predictor,
        architecture=architecture,
        device=torch.device("cpu"),
    )
    return bundle_obj
