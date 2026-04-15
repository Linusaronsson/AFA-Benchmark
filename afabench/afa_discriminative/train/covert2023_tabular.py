import gc
import logging
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torchrl.modules import MLP

from afabench.afa_discriminative.afa_methods import (
    Covert2023AFAMethod,
    GreedyDynamicSelection,
)
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.models import (
    GreedyAFAClassifier,  # noqa: TC001
)
from afabench.afa_discriminative.utils import (
    MaskLayer,
    afa_discriminative_training_prep,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import Covert2023TrainingConfig
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


def train_tabular(cfg: Covert2023TrainingConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    if cfg.smoke_test:
        cfg.nepochs = 1
        cfg.patience = 1

    train_dataset, val_dataset, initializer, unmasker, class_weights = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
            seed=cfg.seed,
        )
    )
    assert class_weights is not None
    class_weights = class_weights.to(device)

    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset, val_dataset, cfg.batch_size
    )

    # Predictor network
    predictor, _ = load_bundle(
        Path(cfg.pretrained_model_bundle_path),
        map_location=device,
    )
    classifier_bundle = cast(
        "GreedyAFAClassifier",
        cast("object", predictor),
    )
    predictor = classifier_bundle.predictor.to(device)
    n_selections = unmasker.get_n_selections(torch.Size([d_in]))
    # assert n_selections == d_in

    selector = MLP(
        in_features=d_in * 2,
        # out_features=d_in,
        out_features=n_selections,
        num_cells=cfg.hidden_units,
        activation_class=getattr(nn, cfg.activation),
        dropout=cfg.dropout,
    ).to(device)

    # GDFS
    mask_layer = MaskLayer(append=True)
    gdfs = GreedyDynamicSelection(
        selector=selector,
        predictor=predictor,
        mask_layer=mask_layer,
        initializer=initializer,
        unmasker=unmasker,
    ).to(device)

    feature_costs = train_dataset.get_feature_acquisition_costs()

    gdfs.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.patience,
        verbose=True,
        feature_costs=feature_costs.to(device),
    )

    # Build final method
    afa_method = Covert2023AFAMethod(
        selector=gdfs.selector.cpu(),
        predictor=gdfs.predictor.cpu(),
        device=torch.device("cpu"),
        selector_hidden_layers=cfg.hidden_units,
        predictor_hidden_layers=cfg.hidden_units,
        dropout=cfg.dropout,
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
        selection_costs=unmasker.get_selection_costs(feature_costs),
        n_selections=n_selections,
    )
    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"Covert2023 method saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
