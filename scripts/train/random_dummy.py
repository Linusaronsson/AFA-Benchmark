import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.afa_methods import RandomDummyAFAMethod
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    RandomDummyTrainConfig,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    initialize_wandb_run,
    set_seed,
)
from afabench.eval.eval import eval_afa_method

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/random_dummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cast(
                "dict[str,Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="pretraining",
            tags=["random_dummy"],
        )
    else:
        run = None

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        # Because this method does not train, smoke test is no different

    train_dataset, dataset_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))

    assert len(train_dataset.label_shape) == 1, "Only 1D labels supported"

    afa_method = RandomDummyAFAMethod(
        device=torch.device("cpu"),
        n_classes=train_dataset.label_shape.numel(),
        prob_select_0=0.0
        if cfg.soft_budget_param is None
        else cfg.soft_budget_param,
    )

    # Create initializer
    initializer = get_afa_initializer_from_config(cfg.initializer)

    # Create unmasker
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)

    # Check that everything works together by doing some evaluation
    eval_afa_method(
        afa_action_fn=afa_method.act,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=unmasker.get_n_selections(
            train_dataset.feature_shape
        ),
        afa_initialize_fn=initializer.initialize,
        dataset=train_dataset,
        external_afa_predict_fn=None,
        builtin_afa_predict_fn=afa_method.predict,
        only_n_samples=100,
        batch_size=10,
    )

    # Save method as a bundle
    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={
            "dataset_class_name": dataset_manifest["class_name"],
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            # "split_idx": dataset_metadata["split_idx"],
            "seed": cfg.seed,
            "soft_budget_param": cfg.soft_budget_param,
            "hard_budget": cfg.hard_budget,
            "initializer_class_name": cfg.initializer.class_name,
            "unmasker_class_name": cfg.unmasker.class_name,
        },
    )

    log.info(f"RandomDummy method saved to {cfg.save_path}")

    if run is not None:
        run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
