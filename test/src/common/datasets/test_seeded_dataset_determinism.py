"""
A seeded dataset generates the same data from the same seed.

A real run regenerates the datasets a smoke test already generated in its
own output root, so both must see the same data.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
import yaml

from afabench.core.registry import get_class

if TYPE_CHECKING:
    from afabench.core.types import AFADataset

DATASET_CONFIGS = (
    Path(__file__).parents[4] / "extra/conf/scripts/dataset_generation/dataset"
)
# Small enough to keep the test fast; the classes take any sample count.
N_SAMPLES = 100


def seeded_dataset_configs() -> list[Any]:
    configs = []
    for path in sorted(DATASET_CONFIGS.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        dataset_class = cast(
            "type[AFADataset]", get_class(config["class_name"])
        )
        if dataset_class.accepts_seed():
            configs.append(pytest.param(config, id=path.stem))
    return configs


@pytest.mark.parametrize("config", seeded_dataset_configs())
def test_the_same_seed_generates_the_same_data(config: dict[str, Any]) -> None:
    dataset_class = cast("type[AFADataset]", get_class(config["class_name"]))
    kwargs = config["kwargs"] | {"n_samples": N_SAMPLES, "seed": 42}

    features, labels = dataset_class(**kwargs).get_all_data()
    again_features, again_labels = dataset_class(**kwargs).get_all_data()

    assert torch.equal(features, again_features)
    assert torch.equal(labels, again_labels)
