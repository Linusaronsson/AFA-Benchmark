import hydra
import wandb
import torch
import logging
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from collections import Counter
from tempfile import NamedTemporaryFile

from afabench.eval.utils import plot_metrics
from afabench.eval.hard_budget import eval_afa_method
from afabench.common.config_classes import EvalConfig
from afabench.common.utils import load_dataset_artifact, set_seed
from afabench.common.afa_uncoverings import (
    one_based_index_uncover_fn,
    get_image_patch_uncover_fn,
)
from afabench.common.registry import (
    get_afa_classifier_class,
    get_afa_method_class,
)

from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
    AFAPredictFn,
)


def load_trained_method_artifacts(
    artifact_name: str, device: torch.device | None = None
) -> tuple[
    AFADataset,
    AFADataset,
    AFADataset,
    AFAMethod,
    dict[str, Any],  # method metadata
]:
    """Load a trained afa method and the dataset it was trained on, from a WandB artifact."""
    if device is None:
        device = torch.device("cpu")
    trained_method_artifact = wandb.use_artifact(
        artifact_name, type="trained_method"
    )
    trained_method_artifact_dir = Path(trained_method_artifact.download())
    method_class = get_afa_method_class(
        trained_method_artifact.metadata["method_type"]
    )
    log.debug(
        f"Loading trained AFA method of class {
            method_class.__name__
        } from artifact {artifact_name}"
    )
    method = method_class.load(trained_method_artifact_dir, device=device)

    # Load the dataset that the method was trained on
    train_dataset, val_dataset, test_dataset, _ = load_dataset_artifact(
        trained_method_artifact.metadata["dataset_artifact_name"]
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        method,
        trained_method_artifact.metadata,
    )


def load_trained_classifier_artifact(
    artifact_name: str, device: torch.device | None = None
) -> tuple[AFAClassifier, dict[str, Any]]:  # classifier metadata
    """Load a trained masked classifier from a WandB artifact."""
    if device is None:
        device = torch.device("cpu")

    trained_classifier_artifact = wandb.use_artifact(
        artifact_name, type="trained_classifier"
    )
    trained_classifier_artifact_dir = Path(
        trained_classifier_artifact.download()
    )
    classifier_class_name = trained_classifier_artifact.metadata[
        "classifier_class_name"
    ]
    classifier_class = get_afa_classifier_class(classifier_class_name)
    log.debug(
        f"Loading trained classifier of class {
            classifier_class_name
        } from artifact {artifact_name}"
    )
    classifier = classifier_class.load(
        trained_classifier_artifact_dir / "classifier.pt", device=device
    )

    return classifier, trained_classifier_artifact.metadata


def validate_artifacts(
    trained_method_artifact_name: str,
    trained_classifier_artifact_name: str,
) -> None:
    """
    Validate that the trained method and classifier artifacts are compatible.

    They should have been trained on the same dataset.
    """
    method_artifact = wandb.use_artifact(
        trained_method_artifact_name, type="trained_method"
    )
    classifier_artifact = wandb.use_artifact(
        trained_classifier_artifact_name, type="trained_classifier"
    )

    classifier_run = classifier_artifact.logged_by()
    assert classifier_run is not None

    assert (
        method_artifact.metadata["dataset_artifact_name"]
        == classifier_run.config["dataset_artifact_name"]
    ), (
        f"The trained method artifact {
            trained_method_artifact_name
        } and the trained classifier artifact {
            trained_classifier_artifact_name
        } "
        "should have been trained on the same dataset, but they are not."
    )

    log.debug(
        f"Method and classifier artifacts are compatible and trained on the same dataset: {
            method_artifact.metadata['dataset_artifact_name']
        }"
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:  # noqa: PLR0915
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load trained afa method and dataset from artifacts
    _train_dataset_, val_dataset, test_dataset, afa_method, method_metadata = (
        load_trained_method_artifacts(
            cfg.trained_method_artifact_name, device=torch.device(cfg.device)
        )
    )
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

    # Load a classifier if it was specified
    if cfg.trained_classifier_artifact_name:
        validate_artifacts(
            cfg.trained_method_artifact_name,
            cfg.trained_classifier_artifact_name,
        )
        afa_predict_fn, classifier_metadata = load_trained_classifier_artifact(
            cfg.trained_classifier_artifact_name,
            device=torch.device(cfg.device),
        )
        classifier_type = classifier_metadata["classifier_type"]
        log.info("Loaded external classifier")
    else:
        log.info("Using builtin classifier")
        afa_predict_fn: AFAPredictFn = afa_method.predict
        if method_metadata["method_type"] == "aaco":
            classifier_type = "MaskedMLPClassifier"
        else:
            classifier_type = "builtin"

    # Use the same hard budget during evaluation as during training
    # Note that this can be None, in which case we will use the maximum number of features in the dataset
    # during evaluation
    # Any method can still choose to stop acquiring features earlier than the hard budget
    if hasattr(cfg, "budget") and getattr(cfg, "budget", None) is not None:
        eval_budget = cfg.budget
        assert type(eval_budget) is int
        log.info(f"Using explicitly provided budget: {eval_budget}")
    elif method_metadata.get("budget") is None:
        log.info("Using maximum number of features in the dataset as budget")
        eval_budget = val_dataset.n_features
    else:
        log.info("Using same budget as during training")
        eval_budget = int(method_metadata["budget"])

    modality = getattr(afa_method, "modality", "tabular")
    image_patch_size = getattr(afa_method, "patch_size", 1)
    afa_uncover_fn = None
    if modality == "image":
        x, _ = dataset[0]
        assert x.ndim == 3
        C, H, W = x.shape
        afa_uncover_fn = get_image_patch_uncover_fn(
            image_side_length=H, n_channels=C, patch_size=image_patch_size
        )
        log.info(
            f"Image modality detected, patch size={
                image_patch_size
            }, image_size=({C}, {H}, {W})."
        )
    else:
        log.info(f"Tabular modality detected.")
        afa_uncover_fn = one_based_index_uncover_fn

    # Do the evaluation
    log.info(f"Starting evaluation with budget {eval_budget}")
    metrics = eval_afa_method(
        afa_method.select,
        dataset,
        eval_budget,
        afa_predict_fn,
        only_n_samples=cfg.eval_only_n_samples,
        batch_size=cfg.batch_size,
        device=torch.device(cfg.device),
        afa_uncover_fn=afa_uncover_fn,
        patch_size=image_patch_size,
    )

    # Log early stopping statistics
    # TODO do we need the step info in hard budget evaluation?
    # log.info(f"Average steps taken: {metrics['average_steps']:.2f}")
    # log.info(f"Maximum steps taken: {metrics['actual_steps'].max()}")
    # log.info(f"Minimum steps taken: {metrics['actual_steps'].min()}")
    # stopped_early = (metrics["actual_steps"] < eval_budget).sum()
    # total_samples = len(metrics["actual_steps"])
    # log.info(
    #     f"Samples that stopped early: {stopped_early}/{total_samples} ({
    #         100 * stopped_early / total_samples:.1f
    #     }%)"
    # )

    fig = plot_metrics(metrics)

    # Create action distribution plot - simplified for Plotly compatibility
    action_fig, action_ax = plt.subplots()
    action_data = metrics["action_distribution"].cpu().numpy()
    feature_indices = list(range(len(action_data)))
    action_ax.plot(feature_indices, action_data)
    action_ax.set_xlabel(
        "Patch index" if modality == "image" else "Feature index"
    )
    action_ax.set_ylabel("Action probability")
    action_ax.set_title("Action Distribution")

    # Create steps distribution plot - ultra-simplified for Plotly
    # steps_fig, steps_ax = plt.subplots()
    # actual_steps = metrics["actual_steps"].cpu().numpy()

    # Use Counter to get step distribution

    # step_counts = Counter(actual_steps)
    # steps = sorted(step_counts.keys())
    # counts = [step_counts[step] for step in steps]

    # # Simple bar plot without any fancy features
    # steps_ax.bar(steps, counts)
    # steps_ax.set_xlabel("Number of steps taken")
    # steps_ax.set_ylabel("Number of samples")
    # steps_ax.set_title(
    #     f"Steps Distribution (Mean: {float(metrics['average_steps']):.1f})"
    # )

    run.log(
        {
            "metrics_plot": fig,
            "action_plot": action_fig,
            # "steps_distribution_plot": steps_fig,
            # "average_steps": metrics["average_steps"],
            # "early_stopping_rate": float(stopped_early) / total_samples,
        }
    )

    # Save results as wandb artifact
    eval_results_artifact = wandb.Artifact(
        name=f"{cfg.trained_method_artifact_name.split(':')[0]}-{
            cfg.trained_classifier_artifact_name.split(':')[0]
            if cfg.trained_classifier_artifact_name
            else 'builtin'
        }",
        type="eval_results",
        metadata={
            "dataset_type": method_metadata["dataset_type"],
            "method_type": method_metadata["method_type"],
            "budget": eval_budget,
            "seed": method_metadata["seed"],
            "classifier_type": classifier_type,
            # "average_steps": float(metrics["average_steps"]),
            # "early_stopping_rate": float(stopped_early) / total_samples,
            # "supports_early_stopping": True,
        },
    )
    with NamedTemporaryFile("w", delete=False) as f:
        metrics_save_path = Path(f.name)
        torch.save(metrics, metrics_save_path)
    eval_results_artifact.add_file(str(metrics_save_path), name="metrics.pt")
    run.log_artifact(
        eval_results_artifact, aliases=cfg.output_artifact_aliases
    )
    run.finish()


if __name__ == "__main__":
    main()
