import json
import wandb
import hydra
import torch
import logging

from pathlib import Path
from omegaconf import OmegaConf

from afabench.common.config_classes import SoftEvalConfig
from afabench.eval.soft_budget import eval_soft_budget_afa_method


from afabench.common.utils import (
    load_trained_method,
    load_trained_classifier,
    validate_method_classifier_compatibility,
    set_seed,
    save_artifact,
    get_artifact_path,
)

from afabench.common.afa_uncoverings import (
    one_based_index_uncover_fn,
    get_image_patch_uncover_fn,
)


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/soft_eval",
    config_name="config",
)
def main(cfg: SoftEvalConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # Optional wandb logging
    # run = wandb.init(
    #     job_type="evaluation",
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     dir="extra/wandb",
    # )

    # log.info(f"W&B run initialized: {run.name} ({run.id})")
    # log.info(f"W&B run URL: {run.url}")

    # Load trained method from filesystem
    _, val_dataset, test_dataset, afa_method, method_metadata = (
        load_trained_method(
            f"{cfg.trained_method_artifact_name}_{cfg.experiment_id}"
            if cfg.experiment_id
            else cfg.trained_method_artifact_name,
            device=torch.device(cfg.device),
        )
    )

    # Select dataset split
    if cfg.dataset_split == "validation":
        dataset = val_dataset
    elif cfg.dataset_split == "testing":
        dataset = test_dataset
    else:
        msg = f"cfg.dataset_split should either be 'validation' or 'testing', not {
            cfg.dataset_split
        }"
        raise ValueError(msg)
    log.info("Loaded trained AFA method and dataset from artifacts")

    # Load external classifier if specified
    external_afa_predict_fn = None
    if cfg.trained_classifier_artifact_name:
        classifier, _ = load_trained_classifier(
            cfg.trained_classifier_artifact_name,
            device=torch.device(cfg.device),
        )
        external_afa_predict_fn = classifier
        log.info("Loaded external classifier")
    else:
        log.info("No external classifier; using builtin predictor")

    # Some methods need to have the cost parameter set during evaluation
    if hasattr(afa_method, "set_cost_param"):
        assert cfg.cost_param is not None, (
            "cfg.cost_param should be set for methods that need to set the cost parameter during evaluation"
        )
        afa_method.set_cost_param(cfg.cost_param)

    # Do the evaluation
    log.info(
        f"Starting evaluation with soft budget, batch size {cfg.batch_size}"
    )
    modality = getattr(afa_method, "modality", "tabular")
    # is_image = modality == "image"
    image_mask_width = getattr(afa_method, "mask_width", None)
    image_patch_size = getattr(afa_method, "patch_size", 1)
    n_patches = getattr(afa_method, "n_patches", 1)
    uncover_fn = None
    if modality == "image":
        x, _ = dataset[0]
        assert x.ndim == 3
        C, H, W = x.shape
        uncover_fn = get_image_patch_uncover_fn(
            image_side_length=H, n_channels=C, patch_size=image_patch_size
        )
    else:
        uncover_fn = one_based_index_uncover_fn

    df_eval = eval_soft_budget_afa_method(
        afa_select_fn=afa_method.select,
        dataset=dataset,
        external_afa_predict_fn=external_afa_predict_fn,
        afa_uncover_fn=uncover_fn,
        builtin_afa_predict_fn=afa_method.predict
        if afa_method.has_builtin_classifier
        else None,
        only_n_samples=cfg.eval_only_n_samples,
        device=torch.device(cfg.device),
        batch_size=cfg.batch_size,
        patch_size=image_patch_size,
    )
    # Add columns to conform to expected format (snake_case)
    df_eval["method"] = method_metadata["method_type"]
    df_eval["training_seed"] = method_metadata["seed"]
    cost_param = afa_method.cost_param
    assert cost_param is not None, (
        "Cost parameter should not be None for soft budget methods"
    )
    df_eval["cost_parameter"] = cost_param
    df_eval["dataset"] = method_metadata["dataset_type"]

    # Log to wandb for debugging purposes
    # run.log({"soft_eval_df": wandb.Table(dataframe=df_eval)})

    # Save results to filesystem
    method_base = cfg.trained_method_artifact_name.split(":")[0]
    if method_base.startswith("train_"):
        method_base = method_base[6:]  # Remove "train_" prefix

    classifier_name = (
        cfg.trained_classifier_artifact_name.split(":")[0]
        if cfg.trained_classifier_artifact_name
        else "builtin"
    )
    if "/" in classifier_name:
        classifier_name = classifier_name.split("/")[-1]

    # Construct eval artifact path
    eval_identifier = (
        f"{method_base}_costparam_{cfg.cost_param}_{classifier_name}"
    )
    artifact_dir = get_artifact_path(
        "trained_method",  # Store under method directory
        f"{method_metadata['method_type']}/{eval_identifier}",
        Path("extra"),
    )

    # Create eval subdirectory
    eval_dir = artifact_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = eval_dir / "soft_eval_data.csv"
    df_eval.to_csv(csv_path, index=False)

    # Save metadata
    eval_metadata = {
        "dataset_type": method_metadata["dataset_type"],
        "method_type": method_metadata["method_type"],
        "seed": method_metadata["seed"],
        "cost_param": cfg.cost_param,
        "eval_type": "soft_budget",
    }

    with open(eval_dir / "metadata.json", "w") as f:
        json.dump(eval_metadata, f, indent=2)

    log.info(f"Evaluation results saved to: {csv_path}")

    # if run:
    #     run.finish()


if __name__ == "__main__":
    main()
