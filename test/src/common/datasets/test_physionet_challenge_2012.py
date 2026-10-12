"""The Challenge 2012 rename exposes only the new dataset identity."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException

from afabench.core.registry import get_class
from afabench.datasets import PhysionetChallenge2012Dataset
from afabench.datasets import config as _config  # noqa: F401
from afabench.datasets.aliases import DATASET_KEY_ALIASES

REPO_ROOT = Path(__file__).parents[4]


def test_dataset_identity_has_no_old_class_alias() -> None:
    assert get_class("PhysionetChallenge2012Dataset") is (
        PhysionetChallenge2012Dataset
    )
    assert DATASET_KEY_ALIASES["PhysionetChallenge2012Dataset"] == (
        "physionet_challenge_2012"
    )
    assert "PhysionetDataset" not in DATASET_KEY_ALIASES
    with pytest.raises(ValueError, match="Unknown class: PhysionetDataset"):
        get_class("PhysionetDataset")


def test_generation_config_selects_the_renamed_dataset() -> None:
    directory = REPO_ROOT / "conf/scripts/dataset_generation"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["dataset=physionet_challenge_2012"],
        )
    assert cfg.dataset.class_name == "PhysionetChallenge2012Dataset"
    assert cfg.dataset.kwargs.root == (
        "data/physionet_challenge_2012/physionet.csv"
    )


def test_generation_rejects_the_old_dataset_key() -> None:
    directory = REPO_ROOT / "conf/scripts/dataset_generation"
    with (
        initialize_config_dir(config_dir=str(directory), version_base=None),
        pytest.raises(MissingConfigException, match="dataset/physionet"),
    ):
        compose(config_name="config", overrides=["dataset=physionet"])
