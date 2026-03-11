import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import CubeNMAROracleTrainConfig
from afabench.common.cube_nm_ar_oracle_method import CubeNMAROracleMethod
from afabench.common.datasets.datasets import CubeNMARDataset
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.cube_nm_ar_unmasker import CubeNMARUnmasker
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import initialize_wandb_run, set_seed
from afabench.eval.eval import eval_afa_method

if TYPE_CHECKING:
    from afabench.common.custom_types import AFAClassifier, AFADataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/cube_nm_ar_oracle",
    config_name="config",
)
def main(cfg: CubeNMAROracleTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cast(
                "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="pretraining",
            tags=["cube_nm_ar_oracle"],
        )
    else:
        run = None

    train_dataset, dataset_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    if not isinstance(train_dataset, CubeNMARDataset):
        msg = (
            "CubeNMAROracleMethod only supports CubeNMARDataset, got "
            f"{type(train_dataset).__name__}."
        )
        raise TypeError(msg)

    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
    if not isinstance(unmasker, CubeNMARUnmasker):
        msg = (
            "CubeNMAROracleMethod expects CubeNMARUnmasker, got "
            f"{type(unmasker).__name__}."
        )
        raise TypeError(msg)

    selection_costs = unmasker.get_selection_costs(
        train_dataset.get_feature_acquisition_costs()
    )

    method = CubeNMAROracleMethod(
        n_contexts=train_dataset.n_contexts,
        n_safe_contexts=train_dataset.n_safe_contexts,
        n_hint_features=train_dataset.n_hint_features,
        n_admin_features=train_dataset.n_admin_features,
        block_size=train_dataset.block_size,
        n_classes=train_dataset.label_shape[-1],
        context_action_cost=float(selection_costs[0].item()),
        selectable_feature_costs=selection_costs[1:],
        device=torch.device(cfg.device),
        max_cost=cfg.soft_budget_param,
    )

    initializer = get_afa_initializer_from_config(cfg.initializer)

    external_classifier = None
    if cfg.classifier_bundle_path is not None:
        classifier, _ = load_bundle(
            Path(cfg.classifier_bundle_path),
            device=torch.device(cfg.device),
        )
        external_classifier = cast("AFAClassifier", cast("object", classifier))

    eval_afa_method(
        afa_action_fn=method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=unmasker.get_n_selections(
            train_dataset.feature_shape
        ),
        afa_initialize_fn=initializer.initialize,
        dataset=train_dataset,
        external_afa_predict_fn=(
            external_classifier.__call__
            if external_classifier is not None
            else None
        ),
        builtin_afa_predict_fn=None,
        only_n_samples=100,
        batch_size=10,
        selection_costs=selection_costs.tolist(),
    )

    save_bundle(
        obj=method,
        path=Path(cfg.save_path),
        metadata={
            "dataset_class_name": dataset_manifest["class_name"],
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            "seed": cfg.seed,
            "soft_budget_param": cfg.soft_budget_param,
            "hard_budget": cfg.hard_budget,
            "initializer_class_name": cfg.initializer.class_name,
            "unmasker_class_name": cfg.unmasker.class_name,
            "context_action_cost": float(selection_costs[0].item()),
            "selection_costs": selection_costs.tolist(),
        },
    )

    log.info("CubeNMAR oracle method saved to %s", cfg.save_path)

    if run is not None:
        run.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
