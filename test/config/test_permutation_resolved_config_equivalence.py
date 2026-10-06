"""
Temporary: prove the training contract port keeps permutation's resolved
hyperparameters. Delete this file once it passes (issue #52).

Values below were captured by composing the pre-port config tree (13
`experiment/<dataset_key>.yaml` files, one per dataset key permutation
supports) for every dataset key, with `hard_budget` overridden on the
command line the way the pipeline always does.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import afabench.components.methods.static.pt.config  # noqa: F401  registers train_permutation

REPO_ROOT = Path(__file__).parents[2]
CONFIG_DIR = REPO_ROOT / "extra/conf/scripts/train_method/permutation"

_PRE_PORT_HYPERPARAMETERS = {
    "batch_size": 128,
    "selector": {
        "lr": 0.001,
        "nepochs": 250,
        "num_cells": [128, 128],
        "patience": 5,
    },
    "classifier": {"lr": 0.001, "nepochs": 250, "num_cells": [128, 128]},
}

DATASET_KEYS = [
    "actg",
    "bank_marketing",
    "ckd",
    "cube",
    "cube_nonuniform_costs",
    "cube_without_noise",
    "diabetes",
    "fashion_mnist",
    "miniboone",
    "mnist",
    "physionet",
    "synthetic_mnist",
    "synthetic_mnist_without_noise",
]


@pytest.mark.parametrize("dataset_key", DATASET_KEYS)
def test_resolved_hyperparameters_match_pre_port_values(
    dataset_key: str,
) -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_DIR)
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "initializer=cold",
                "unmasker=direct",
                f"dataset_key={dataset_key}",
                "train_dataset_bundle_path=x",
                "val_dataset_bundle_path=y",
                "classifier_bundle_path=z",
                "save_path=s",
                "hard_budget=99",
                "soft_budget_param=null",
                "device=cpu",
                "seed=1",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    hyperparameters = {
        key: resolved[key] for key in _PRE_PORT_HYPERPARAMETERS
    }

    assert hyperparameters == _PRE_PORT_HYPERPARAMETERS
