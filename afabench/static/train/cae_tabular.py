import gc
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchrl.modules import MLP

from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.utils import afa_discriminative_training_prep
from afabench.common.bundle import save_bundle
from afabench.common.config_classes import CAETrainingConfig
from afabench.common.utils import set_seed
from afabench.static.models import BaseModel
from afabench.static.static_methods import (
    ConcreteMask,
    DifferentiableSelector,
    StaticBaseMethod,
)
from afabench.static.utils import transform_dataset

log = logging.getLogger(__name__)


def train_tabular(cfg: CAETrainingConfig) -> None:
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")

    train_dataset, val_dataset, _, _, class_weights = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    assert class_weights is not None
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    model = MLP(
        in_features=d_in,
        out_features=d_out,
        num_cells=cfg.selector.num_cells,
        activation_class=nn.ReLU,
    )
    selector_layer = ConcreteMask(d_in, cfg.hard_budget)
    diff_selector = DifferentiableSelector(
        model=model,
        selector_layer=selector_layer,
    ).to(device)
    diff_selector.fit(
        train_loader,
        val_loader,
        lr=cfg.selector.lr,
        nepochs=cfg.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.selector.patience,
        verbose=False,
    )

    logits = selector_layer.logits.cpu().data.numpy()
    ranked_features = np.sort(logits.argmax(axis=1))

    if len(np.unique(ranked_features)) != cfg.hard_budget:
        print(
            f"{len(np.unique(ranked_features))} selected instead of {
                cfg.hard_budget
            }, appending extras"
        )
    num_extras = cfg.hard_budget - len(np.unique(ranked_features))
    remaining_features = np.setdiff1d(np.arange(d_in), ranked_features)
    ranked_features = np.sort(
        np.concatenate(
            [np.unique(ranked_features), remaining_features[:num_extras]]
        )
    )

    predictors: dict[int, nn.Module] = {}
    selected_history: dict[int, list[int]] = {}

    num_features = list(range(1, cfg.hard_budget + 1))
    for num in num_features:
        selected_features = ranked_features[:num]
        selected_history[num] = selected_features.tolist()

        train_subset = transform_dataset(train_dataset, selected_features)
        val_subset = transform_dataset(val_dataset, selected_features)

        train_subset_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_subset_loader = DataLoader(
            val_subset, batch_size=cfg.batch_size, pin_memory=True
        )

        model = MLP(
            in_features=num,
            out_features=d_out,
            num_cells=cfg.classifier.num_cells,
            activation_class=nn.ReLU,
        )
        predictor = BaseModel(model).to(device)
        predictor.fit(
            train_subset_loader,
            val_subset_loader,
            lr=cfg.classifier.lr,
            nepochs=cfg.classifier.nepochs,
            loss_fn=nn.CrossEntropyLoss(weight=class_weights),
            verbose=False,
        )

        predictors[num] = model

    static_method = StaticBaseMethod(selected_history, predictors, device)

    save_bundle(
        obj=static_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    log.info(f"CAE method saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
