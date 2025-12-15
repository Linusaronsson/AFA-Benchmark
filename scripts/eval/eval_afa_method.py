import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import OmegaConf

from afabench.afa_oracle.afa_methods import AACOAFAMethod
from afabench.common.bundle import load_bundle
from afabench.common.config_classes import (
    EvalConfig,
    InitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    set_seed,
)
from afabench.eval.eval import eval_afa_method

log = logging.getLogger(__name__)


def load(
    method_bundle_path: Path,
    unmasker_cfg: UnmaskerConfig,
    initializer_cfg: InitializerConfig,
    dataset_bundle_path: Path,
    classifier_bundle_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[
    AFAMethod,
    AFAUnmasker,
    AFAInitializer,
    AFADataset,
    AFAClassifier | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
]:
    # Load method
    device = torch.device("cpu") if device is None else device
    method, method_manifest = load_bundle(
        method_bundle_path,
        device=device,
    )
    method = cast("AFAMethod", cast("object", method))
    log.info(f"Loaded AFA method from {method_bundle_path}")

    # Load unmasker
    unmasker: AFAUnmasker = get_afa_unmasker_from_config(unmasker_cfg)
    log.info(f"Loaded {unmasker_cfg.class_name}")

    # Load initializer
    initializer: AFAInitializer = get_afa_initializer_from_config(
        initializer_cfg
    )
    log.info(f"Loaded {initializer_cfg.class_name} initializer")

    # Load dataset
    dataset, dataset_manifest = load_bundle(dataset_bundle_path)
    dataset = cast("AFADataset", cast("object", dataset))
    log.info(f"Loaded dataset from {dataset_bundle_path}")

    # Load external classifier if specified
    if classifier_bundle_path is not None:
        classifier, classifier_manifest = load_bundle(
            classifier_bundle_path,
            device=device,
        )
        classifier = cast("AFAClassifier", cast("object", classifier))
        log.info(f"Loaded external classifier from {classifier_bundle_path}.")
        classifier_metadata = classifier_manifest["metadata"]
    else:
        classifier = None
        classifier_metadata = None
        log.info("No external classifier provided; using builtin classifier.")

    return (
        method,
        unmasker,
        initializer,
        dataset,
        classifier,
        method_manifest["metadata"],
        dataset_manifest["metadata"],
        classifier_metadata,
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = wandb.init(
            job_type="evaluation",
            config=cast(
                "dict[str, Any]",
                OmegaConf.to_container(cfg, resolve=True),
            ),
            dir="extra/wandb",
        )
        log.info(f"W&B run initialized: {run.name} ({run.id})")
        log.info(f"W&B run URL: {run.url}")
    else:
        run = None

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.eval_only_n_samples = 10
        cfg.batch_size = 2

    # Load everything
    (
        afa_method,
        unmasker,
        initializer,
        dataset,
        external_classifier,
        method_metadata,
        dataset_metadata,
        external_classifier_metadata,
    ) = load(
        method_bundle_path=Path(cfg.method_bundle_path),
        unmasker_cfg=cfg.unmasker,
        initializer_cfg=cfg.initializer,
        dataset_bundle_path=Path(cfg.dataset_bundle_path),
        classifier_bundle_path=(
            Path(cfg.classifier_bundle_path)
            if cfg.classifier_bundle_path is not None
            else None
        ),
        device=torch.device(cfg.device),
    )

    # Set the seed of everything
    afa_method.set_seed(cfg.seed)
    unmasker.set_seed(cfg.seed)
    initializer.set_seed(cfg.seed)

    if cfg.hard_budget is not None:
        hard_budget_str = f"hard budget {cfg.hard_budget}"
    else:
        hard_budget_str = "no hard budget"
    log.info(
        f"Starting evaluation with batch size {cfg.batch_size} and {
            hard_budget_str
        }."
    )

    if isinstance(afa_method, AACOAFAMethod):
        afa_method.aaco_oracle.set_classifier(
            external_classifier
            if external_classifier is not None
            else afa_method.aaco_oracle.classifier
        )

    afa_method = cast(
        "AFAMethod", afa_method
    )  # TODO: remove once AACOAFAMethod implements AFAMethod
    df_eval = eval_afa_method(
        afa_select_fn=afa_method.select,
        afa_unmask_fn=unmasker.unmask,
        n_selection_choices=unmasker.get_n_selections(
            feature_shape=dataset.feature_shape
        ),
        afa_initialize_fn=initializer.initialize,
        dataset=dataset,
        external_afa_predict_fn=external_classifier.__call__
        if external_classifier is not None
        else None,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
        selection_budget=cfg.hard_budget,
        batch_size=cfg.batch_size,
    )

    # Add eval_seed and eval_hard_budget to dataframe
    df_eval["eval_seed"] = cfg.seed
    df_eval["eval_hard_budget"] = (
        cfg.hard_budget if cfg.hard_budget is not None else float("nan")
    )

    # Save CSV directly
    csv_path = Path(cfg.save_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(csv_path, index=False)
    log.info(f"Saved evaluation data to CSV at: {csv_path}")

    log.info(f"Evaluation results saved to: {cfg.save_path}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
