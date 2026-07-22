import gc
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
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
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSPretrainingConfig,
    GDFSTabularArchitectureConfig,
)
from afabench.core.bundle_system.bundle import (
    load_bundle,
    save_bundle,
)
from afabench.core.naming import infer_dataset_key_from_class_name
from afabench.core.utils import (
    get_class_frequencies,
    set_seed,
)

log = logging.getLogger(__name__)


def pretrain_tabular(
    cfg: GDFSPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
) -> None:
    log.debug(cfg)
    assert isinstance(cfg.architecture, GDFSTabularArchitectureConfig)
    assert cfg.device is not None, "device must be configured"
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    if cfg.smoke_test:
        cfg.nepochs = 1
        cfg.patience = 1

    train_dataset, train_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path)
    )
    val_dataset, _ = load_bundle(Path(cfg.val_dataset_bundle_path))

    dataset_name = infer_dataset_key_from_class_name(
        train_manifest["class_name"]
    )
    log.info("Pretraining GDFS predictor on dataset %s", dataset_name)
    _, train_labels = train_dataset.get_all_data()  # pyright: ignore[reportAttributeAccessIssue]
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
    pretrainer = MaskingPretrainer(predictor, mask_layer).to(device)

    pretrainer.fit(
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
        metric_prefix="gdfs_pretrain",
    )

    metadata = {
        "model_type": "GDFSClassifier",
        "dataset_name": dataset_name,
        "pretrain_config": asdict(cfg),
    }
    bundle_obj = GreedyAFAClassifier(
        predictor=predictor,
        architecture=architecture,
        device=torch.device("cpu"),
    )

    save_bundle(
        obj=bundle_obj,
        path=Path(cfg.save_path),
        metadata=metadata,
    )

    log.info(f"GDFS pretrained model saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
