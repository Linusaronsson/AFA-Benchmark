import torch
import wandb
import hydra
import logging

from pathlib import Path
from time import strftime
from torch.utils.data import random_split

from afabench.common.custom_types import AFADataset
from afabench.common.registry import get_afa_dataset_class
from afabench.common.utils import save_artifact, get_artifact_path
from afabench.common.config_classes import (
    DatasetGenerationConfig,
    SplitRatioConfig,
)


log = logging.getLogger(__name__)


def generate_and_save_split(
    dataset_class: type[AFADataset],
    dataset_type: str,
    split_idx: int,
    split_ratio: SplitRatioConfig,
    seed: int,
    base_dir: Path = Path("extra"),
    epsilon: float = 1e-8,
    **dataset_kwargs,
):
    """Generate and save a single train/val/test split."""
    # Generate full dataset
    dataset_kwargs["seed"] = seed
    dataset = dataset_class(**dataset_kwargs)
    dataset.generate_data()

    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(split_ratio.train * total_size)
    val_size = int(split_ratio.val * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset,  # pyright: ignore[reportArgumentType]
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create split datasets
    train_dataset = dataset_class(**dataset_kwargs)
    train_dataset.features = dataset.features[train_subset.indices]
    train_dataset.labels = dataset.labels[train_subset.indices]

    val_dataset = dataset_class(**dataset_kwargs)
    val_dataset.features = dataset.features[val_subset.indices]
    val_dataset.labels = dataset.labels[val_subset.indices]

    test_dataset = dataset_class(**dataset_kwargs)
    test_dataset.features = dataset.features[test_subset.indices]
    test_dataset.labels = dataset.labels[test_subset.indices]

    # Normalize if needed
    if dataset_kwargs.get("normalize", False):
        mean = train_dataset.features.mean(dim=0)
        std = train_dataset.features.std(dim=0)
        for ds in (train_dataset, val_dataset, test_dataset):
            ds.features = (ds.features - mean) / (std + epsilon)

    # Save to temporary location
    temp_dir = (
        Path("tmp")
        / f"{dataset_type}_split_{split_idx}_{strftime('%Y%m%d_%H%M%S')}"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_path = temp_dir / "train.pt"
    val_path = temp_dir / "val.pt"
    test_path = temp_dir / "test.pt"

    train_dataset.save(train_path)
    val_dataset.save(val_path)
    test_dataset.save(test_path)

    # Prepare metadata
    metadata = dataset_kwargs | {
        "dataset_type": dataset_type,
        "split_idx": split_idx,
        "seed": seed,
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to filesystem using new artifact system
    artifact_name = f"{dataset_type}/{dataset_type}_split_{split_idx}"
    artifact_dir = get_artifact_path("dataset", artifact_name, base_dir)

    save_artifact(
        artifact_dir=artifact_dir,
        files={
            "train.pt": train_path,
            "val.pt": val_path,
            "test.pt": test_path,
        },
        metadata=metadata,
    )

    # Clean up temp files
    import shutil

    shutil.rmtree(temp_dir)

    log.info(f"Saved {dataset_type} split {split_idx} to {artifact_dir}")
    log.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {
            len(test_dataset)
        }"
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    # Optional: still init wandb for logging metrics
    # run = wandb.run or wandb.init(job_type="data_generation", dir="extra/wandb")

    for seed in cfg.seeds:
        for split_idx in cfg.split_idx:
            dataset_class = get_afa_dataset_class(cfg.dataset.type)
            generate_and_save_split(
                dataset_class=dataset_class,
                dataset_type=cfg.dataset.type,
                split_idx=split_idx,
                split_ratio=cfg.split_ratio,
                seed=seed,
                base_dir=Path(cfg.data_dir),
                **cfg.dataset.kwargs,
            )

    # if run:
    #     run.finish()


if __name__ == "__main__":
    main()
