from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import afabench.components.classifiers.config  # noqa: F401

if TYPE_CHECKING:
    from afabench.components.classifiers.config import (
        TrainMaskedMLPClassifierConfig,
    )


@pytest.mark.parametrize(
    ("dataset_key", "epochs", "batch_size", "dropout"),
    [
        ("mnist", 100, 64, 0.2),
        ("fashion_mnist", 100, 64, 0.2),
        ("synthetic_mnist", 100, 64, 0.2),
        ("synthetic_mnist_without_noise", 100, 64, 0.2),
        ("cube", 2000, 128, 0.1),
    ],
)
def test_classifier_config_composes_for_workflow_dataset_keys(
    dataset_key: str,
    epochs: int,
    batch_size: int,
    dropout: float,
) -> None:
    config_dir = Path(
        "conf/scripts/train_classifier/masked_mlp_classifier"
    ).resolve()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "train_dataset_path=train.bundle",
                "val_dataset_path=val.bundle",
                "save_path=classifier.bundle",
                "initializer=cold",
                "unmasker=direct",
                f"dataset_override@_global_={dataset_key}",
            ],
        )

    classifier_cfg = cast(
        "TrainMaskedMLPClassifierConfig", OmegaConf.to_object(cfg)
    )

    assert classifier_cfg.epochs == epochs
    assert classifier_cfg.batch_size == batch_size
    assert classifier_cfg.dropout == dropout
    assert classifier_cfg.min_masking_probability == 0.0
    assert classifier_cfg.max_masking_probability == 0.9
    assert classifier_cfg.eval_only_n_samples is None
    assert classifier_cfg.lr == 0.001
    assert classifier_cfg.num_cells == [128, 128]
