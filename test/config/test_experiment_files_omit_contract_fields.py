"""
No experiment file of a ported method may set a contract field.

See docs/adr/0001-training-contract-as-library.md: the pipeline always
overrides contract fields on the command line, so a value in an
experiment file is dead in the pipeline and misleading to a developer
running the script by hand. `PORTED_METHODS` grows as each port lands;
see docs/training_contract_inventory.md for the methods still to port.
"""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.training.contract import TrainingContract

REPO_ROOT = Path(__file__).parents[2]
CONTRACT_FIELDS = {f.name for f in fields(TrainingContract)}

PORTED_METHODS = [
    "train_method/permutation",
]


def _experiment_files() -> list[Path]:
    return sorted(
        experiment_file
        for method in PORTED_METHODS
        for experiment_file in (
            REPO_ROOT / "extra/conf/scripts" / method / "experiment"
        ).glob("*.yaml")
    )


@pytest.mark.parametrize(
    "experiment_file",
    _experiment_files(),
    ids=[str(f.relative_to(REPO_ROOT)) for f in _experiment_files()],
)
def test_experiment_file_does_not_set_a_contract_field(
    experiment_file: Path,
) -> None:
    content = OmegaConf.load(experiment_file)
    keys = set(content.keys()) if OmegaConf.is_dict(content) else set()

    assert keys & CONTRACT_FIELDS == set()
