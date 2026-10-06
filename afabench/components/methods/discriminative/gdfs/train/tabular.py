import logging
from collections.abc import Callable

import torch
from torch import nn
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
    GreedyDynamicSelection,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSTabularArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.core.utils import get_class_frequencies
from afabench.training.inputs import TrainingInputs

log = logging.getLogger(__name__)


def train_tabular(
    cfg: GDFSTrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: TrainingInputs,
) -> GDFSAFAMethod:
    log.debug(cfg)
    assert isinstance(cfg.architecture, GDFSTabularArchitectureConfig)
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    assert cfg.pretrained_model_bundle_path is not None, (
        "pretrained_model_bundle_path must be configured"
    )
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    train_dataset = inputs.train_dataset()
    val_dataset = inputs.val_dataset()
    initializer = inputs.initializer()
    unmasker = inputs.unmasker()
    _, train_labels = train_dataset.get_all_data()
    class_weights = 1 / get_class_frequencies(train_labels)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset,
        val_dataset,
        cfg.batch_size,
        smoke_test=cfg.smoke_test,
    )
    classifier_bundle = inputs.pretrained_model(GreedyAFAClassifier)
    predictor = classifier_bundle.predictor.to(device)
    n_selections = unmasker.get_n_selections(train_dataset.feature_shape)
    selector = MLP(
        in_features=d_in * 2,
        out_features=n_selections,
        num_cells=cfg.architecture.hidden_units,
        activation_class=getattr(nn, cfg.architecture.activation),
        dropout=cfg.architecture.dropout,
    ).to(device)
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
        temp_steps=1 if cfg.smoke_test else 5,
        verbose=True,
        feature_costs=feature_costs.to(device),
        metric_logger=metric_logger,
        metric_prefix="gdfs",
    )
    afa_method = GDFSAFAMethod(
        selector=gdfs.selector.cpu(),
        predictor=gdfs.predictor.cpu(),
        device=torch.device("cpu"),
        selector_hidden_layers=cfg.architecture.hidden_units,
        predictor_hidden_layers=cfg.architecture.hidden_units,
        dropout=cfg.architecture.dropout,
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
        selection_costs=unmasker.get_selection_costs(feature_costs),
        n_selections=n_selections,
    )
    return afa_method
