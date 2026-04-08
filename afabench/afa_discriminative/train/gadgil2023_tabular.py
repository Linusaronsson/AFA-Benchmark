import gc
import logging
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torchrl.modules import MLP

from afabench.afa_discriminative.afa_methods import (
    CMIEstimator,
    Gadgil2023AFAMethod,
)
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.afa_discriminative.models import (
    GreedyAFAClassifier,
    NotMIWAE,
    train_notmiwae,
)
from afabench.afa_discriminative.utils import (
    MaskLayer,
    afa_discriminative_training_prep,
    precompute_initial_and_forbidden_masks,
    tie_first_k_linears_by_module,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import Gadgil2023TrainingConfig
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


# def _get_initial_observation_mask(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     initializer: AFAInitializer,
#     feature_shape: torch.Size,
# ) -> torch.Tensor:
#     with torch.no_grad():
#         y_idx = to_class_indices(y)
#         init_mask_bool = initializer.initialize(
#             features=x,
#             label=y_idx,
#             feature_shape=feature_shape,
#         )
#         forbidden_feat = initializer.get_training_forbidden_mask(
#             init_mask_bool
#         )
#         init_mask_bool = init_mask_bool & ~forbidden_feat
#         return init_mask_bool.float()


def train_tabular(cfg: Gadgil2023TrainingConfig) -> None:
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
        )
    )
    assert class_weights is not None
    class_weights = class_weights.to(device)

    feature_shape = torch.Size([train_dataset.feature_shape[0]])
    # TODO: different missingness models for train and validation, distribution shift?
    train_obs_mask, train_forbidden_mask = (
        precompute_initial_and_forbidden_masks(
            train_dataset,
            initializer,
            feature_shape,
            device,
        )
    )
    val_obs_mask, val_forbidden_mask = precompute_initial_and_forbidden_masks(
        val_dataset,
        initializer,
        feature_shape,
        device,
    )
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg.batch_size,
        train_observed_mask=train_obs_mask,
        val_observed_mask=val_obs_mask,
        train_forbidden_mask=train_forbidden_mask,
        val_forbidden_mask=val_forbidden_mask,
    )

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

    value_network = MLP(
        in_features=d_in * 2,
        out_features=n_selections,
        num_cells=cfg.hidden_units,
        activation_class=getattr(nn, cfg.activation),
        dropout=cfg.dropout,
    ).to(device)

    tie_first_k_linears_by_module(predictor, value_network, k=2)

    mask_layer = MaskLayer(append=True)

    notmiwae_model = None
    if cfg.ipw_mode == "notmiwae_feature":
        train_x_all, train_y_all = train_dataset.get_all_data()
        val_x_all, val_y_all = val_dataset.get_all_data()
        train_x_all = train_x_all.to(device)
        train_y_all = train_y_all.to(device)
        val_x_all = val_x_all.to(device)
        val_y_all = val_y_all.to(device)

        train_s_all = train_obs_mask.to(dtype=train_x_all.dtype, device=device)
        val_s_all = val_obs_mask.to(dtype=val_x_all.dtype, device=device)

        # train_s_all = _get_initial_observation_mask(
        #     x=train_x_all,
        #     y=train_y_all,
        #     initializer=initializer,
        #     feature_shape=feature_shape,
        # ).to(device)

        # val_s_all = _get_initial_observation_mask(
        #     x=val_x_all,
        #     y=val_y_all,
        #     initializer=initializer,
        #     feature_shape=feature_shape,
        # ).to(device)

        train_x_filled = train_x_all * train_s_all
        val_x_filled = val_x_all * val_s_all

        # TODO: currently not compatible with the CubeNM unmasker
        notmiwae_model = NotMIWAE(
            d_in=d_in,
            n_latent=max(1, d_in - 1),
            n_hidden=128,
            activation=nn.Tanh,
        ).to(device)

        notmiwae_model = train_notmiwae(
            model=notmiwae_model,
            x_train=train_x_filled,
            s_train=train_s_all,
            x_val=val_x_filled,
            s_val=val_s_all,
            lr=1e-3,
            batch_size=128,
            max_iter=30000 if not cfg.smoke_test else 50,
            n_samples=20,
            eval_every=100,
        )

    greedy_cmi_estimator = CMIEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        # initializer=initializer,
        unmasker=unmasker,
        notmiwae_model=notmiwae_model,
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
        ipw_mode=cfg.ipw_mode,
        ipw_min_propensity=cfg.ipw_min_propensity,
        ipw_max_weight=cfg.ipw_max_weight,
        ipw_normalize_weights=cfg.ipw_normalize_weights,
    )

    afa_method = Gadgil2023AFAMethod(
        greedy_cmi_estimator.value_network.cpu(),
        greedy_cmi_estimator.predictor.cpu(),
        device=torch.device("cpu"),
        value_network_hidden_layers=cfg.hidden_units,
        predictor_hidden_layers=cfg.hidden_units,
        dropout=cfg.dropout,
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
        n_selections=n_selections,
        selection_costs=unmasker.get_selection_costs(feature_costs),
    )

    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"Gadgil2023 method saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
