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
    tie_first_k_linears_by_module,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    CMIEstimator,
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMETabularArchitectureConfig,
    DIMETrainingConfig,
)
from afabench.core.utils import get_class_frequencies
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)


def train_tabular(
    cfg: DIMETrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> DIMEAFAMethod:
    log.debug(cfg)
    assert isinstance(cfg.architecture, DIMETabularArchitectureConfig)
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
    value_network = MLP(
        in_features=d_in * 2,
        out_features=n_selections,
        num_cells=cfg.architecture.hidden_units,
        activation_class=getattr(nn, cfg.architecture.activation),
        dropout=cfg.architecture.dropout,
    ).to(device)
    tie_first_k_linears_by_module(predictor, value_network, k=2)
    mask_layer = MaskLayer(append=True)
    greedy_cmi_estimator = CMIEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        initializer=initializer,
        unmasker=unmasker,
    ).to(device)
    feature_costs = train_dataset.get_feature_acquisition_costs()
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        eps=cfg.eps,
        loss_fn=nn.CrossEntropyLoss(reduction="none", weight=class_weights),
        val_loss_fn=None,
        val_loss_mode=None,
        eps_decay=cfg.eps_decay,
        eps_steps=cfg.eps_steps,
        patience=cfg.patience,
        feature_costs=feature_costs.to(device),
        metric_logger=metric_logger,
        metric_prefix="dime",
    )
    afa_method = DIMEAFAMethod(
        greedy_cmi_estimator.value_network.cpu(),
        greedy_cmi_estimator.predictor.cpu(),
        device=torch.device("cpu"),
        value_network_hidden_layers=cfg.architecture.hidden_units,
        predictor_hidden_layers=cfg.architecture.hidden_units,
        dropout=cfg.architecture.dropout,
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
        n_selections=n_selections,
        selection_costs=unmasker.get_selection_costs(feature_costs),
    )
    return afa_method
