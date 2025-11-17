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
    set_seed,
    save_artifact,
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
    method_artifact_name = (
        f"{cfg.trained_method_artifact_name}_{cfg.experiment_id}"
        if cfg.experiment_id
        else cfg.trained_method_artifact_name
    )
    _, val_dataset, test_dataset, afa_method, method_metadata = (
        load_trained_method(
            method_artifact_name,
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
    image_patch_size = getattr(afa_method, "patch_size", 1)
    uncover_fn = None

    if modality == "image":
        x, _ = dataset[0]
        assert x.ndim == 3
        C, H, _ = x.shape
        uncover_fn = get_image_patch_uncover_fn(
            image_side_length=H, n_channels=C, patch_size=image_patch_size
        )
    else:
        uncover_fn = one_based_index_uncover_fn

    # Uncomment the evaluation
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

    # Remove "train_" prefix from method_artifact_name
    eval_artifact_name = method_artifact_name.replace("train_", "")

    # Parse method name (first part before underscore)
    method_name = eval_artifact_name.split("_")[0]

    # Create eval directory: extra/result/{method_name}/eval/{artifact_name}/
    base_dir = Path("extra/result")
    # remove "methodname_" prefix from eval_artifact_name
    eval_artifact_name = eval_artifact_name.split("_", 1)[-1]
    eval_dir = base_dir / method_name / "eval" / eval_artifact_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV directly
    csv_path = eval_dir / "soft_eval_data.csv"
    df_eval.to_csv(csv_path, index=False)
    log.info(f"Saved evaluation data to CSV at: {csv_path}")

    # Save metadata
    eval_metadata = {
        "dataset_type": method_metadata["dataset_type"],
        "method_type": method_metadata["method_type"],
        "seed": method_metadata["seed"],
        "cost_param": cfg.cost_param,
        "eval_type": "soft_budget",
        "dataset_split": cfg.dataset_split,
        "classifier_artifact_name": cfg.trained_classifier_artifact_name,
    }

    with open(eval_dir / "metadata.json", "w") as f:
        json.dump(eval_metadata, f, indent=2)

    log.info(f"Evaluation results saved to: {eval_dir}")

    # if run:
    #     run.finish()


if __name__ == "__main__":
    main()
