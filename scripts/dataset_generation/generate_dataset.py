"""Generate multiple dataset realizations of a dataset, see dataset_generation.md."""

import logging
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.provenance import Split, capture_provenance
from afabench.core.registry import get_class
from afabench.core.types import AFADataset
from afabench.datasets.config import DatasetGenerationConfig, SplitRatioConfig
from afabench.datasets.utils import require_generation_order

log = logging.getLogger(__name__)


def generate_and_save_split(
    dataset_class: type[AFADataset],
    split_ratio: SplitRatioConfig,
    seed_for_split: int,
    save_path: Path,
    dataset_kwargs: dict[str, Any],
    metadata_to_save: dict[str, Any],
    dataset_key: str,
    dataset_realization_index: int,
    resolved_config: dict[str, Any],
) -> None:
    """
    Generate and save a single train/val/test split.

    Args:
        dataset_class: The dataset class to instantiate.
        split_ratio: The ratio for splitting the dataset into train/val/test.
        seed_for_split: Seed used during splitting.
        save_path: Path to save the generated dataset splits. Will create a separate folder for each dataset realization.
        dataset_kwargs: Keyword arguments to pass to the dataset class constructor.
        metadata_to_save: Additional metadata to save alongside the dataset.
        dataset_key: The dataset key, recorded in each bundle's provenance.
        dataset_realization_index: The realization's index, recorded in each bundle's provenance.
        resolved_config: The script's full configuration, recorded in each bundle's provenance.
    """
    # Generate full dataset. Splitting by position below makes each split's
    # generation indices positions in this dataset.
    dataset = dataset_class(**dataset_kwargs)
    require_generation_order(dataset)

    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(split_ratio.train * total_size)
    val_size = int(split_ratio.val * total_size)

    all_indices = list(range(total_size))
    rnd = random.Random(seed_for_split)
    rnd.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]
    test_indices = all_indices[train_size + val_size :]

    train_dataset = dataset.create_subset(train_indices)
    val_dataset = dataset.create_subset(val_indices)
    test_dataset = dataset.create_subset(test_indices)

    # Save splits
    save_path.mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train.bundle"
    val_path = save_path / "val.bundle"
    test_path = save_path / "test.bundle"

    splits: list[Split] = ["train", "val", "test"]
    for obj, path, split in zip(
        [train_dataset, val_dataset, test_dataset],
        [train_path, val_path, test_path],
        splits,
        strict=True,
    ):
        save_bundle(
            obj=obj,
            path=path,
            metadata=metadata_to_save
            | {
                "seed_for_split": seed_for_split,
                "generated_at": datetime.now(UTC).isoformat(),
                "kwargs": dataset_kwargs,
            },
            provenance=capture_provenance(
                stage="dataset_generation",
                resolved_config=resolved_config,
                seed=seed_for_split,
                smoke_test=False,
                device="cpu",
                dataset_key=dataset_key,
                dataset_realization_index=dataset_realization_index,
                split=split,
            ),
        )

    # # Prepare metadata
    # metadata_to_save = metadata_to_save | {
    #     "seed_for_split": seed_for_split,
    #     "generated_at": datetime.now(UTC).isoformat(),
    # }
    # json_data = metadata_to_save | {
    #     "kwargs": dataset_kwargs,
    # }
    # # Save metadata
    # with (save_path / "metadata.json").open("w") as f:
    #     json.dump(json_data, f)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    cfg = cast("DatasetGenerationConfig", OmegaConf.to_object(cfg))
    # The dataset key is the selected `dataset` config, not the class name:
    # several keys (cube, cube_without_noise) share one class.
    dataset_key = HydraConfig.get().runtime.choices["dataset"]
    log.info(f"Generating {cfg.dataset.class_name} to {cfg.save_path}")
    for dataset_realization_index, seed in zip(
        cfg.dataset_realization_indices, cfg.seeds, strict=True
    ):
        dataset_class = cast(
            "type[AFADataset]", get_class(cfg.dataset.class_name)
        )
        if dataset_class.accepts_seed():
            dataset_kwargs = dict(cfg.dataset.kwargs) | {"seed": seed}
        else:
            dataset_kwargs = dict(cfg.dataset.kwargs)
        generate_and_save_split(
            dataset_class=dataset_class,
            split_ratio=cfg.split_ratio,
            # use same instance for splitting as for data generation
            seed_for_split=seed,
            save_path=Path(cfg.save_path) / str(dataset_realization_index),
            dataset_kwargs=dataset_kwargs,
            metadata_to_save={
                "dataset_realization_index": dataset_realization_index,
            },
            dataset_key=dataset_key,
            dataset_realization_index=dataset_realization_index,
            resolved_config=asdict(cfg),
        )
    log.info(
        f"Generated {
            len(cfg.dataset_realization_indices)
        } dataset realizations to {cfg.save_path}"
    )


if __name__ == "__main__":
    main()
