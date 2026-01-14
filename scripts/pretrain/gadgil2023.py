import gc
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Accuracy
from torchrl.modules import MLP

from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.models import (
    GreedyAFAClassifier,
    MaskingPretrainer,
)
from afabench.afa_discriminative.utils import MaskLayer
from afabench.common.bundle import (
    load_bundle,
    save_bundle,
)
from afabench.common.config_classes import Gadgil2023PretrainingConfig
from afabench.common.utils import (
    get_class_frequencies,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023PretrainingConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    train_dataset, train_manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    val_dataset, _ = load_bundle(Path(cfg.val_dataset_bundle_path))

    dataset_name = train_manifest["class_name"].replace("Dataset", "").lower()
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
    hidden_units = cfg.hidden_units
    activation_name: str = cfg.activation
    dropout: float = float(cfg.dropout)

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
        val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        val_loss_mode="max",
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
    )

    metadata = {
        "model_type": "Gadgil2023Classifier",
        "dataset_name": dataset_name,
        "pretrain_config": OmegaConf.to_container(cfg),
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

    log.info(f"Gadgil2023 pretrained model saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
