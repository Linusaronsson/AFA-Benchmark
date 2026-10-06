"""
No experiment file of a ported method may set a contract field.

See docs/adr/0001-training-contract-as-library.md: the pipeline always
overrides contract fields on the command line, so a value in an
experiment file is dead in the pipeline and misleading to a developer
running the script by hand. `KNOWN_VIOLATIONS` lists the unported
methods' pre-existing violations; shrink it as each port lands, and
remove its entry once that method's experiment files are clean.
"""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.training.contract import TrainingContract

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_CONF_ROOT = REPO_ROOT / "extra/conf/scripts"
CONTRACT_FIELDS = {f.name for f in fields(TrainingContract)}

KNOWN_VIOLATIONS: dict[str, set[str]] = {
    "pretrain_model/aaco": {"seed", "device"},
    "train_method/aaco": {"seed", "device"},
    "train_method/cae": {"hard_budget"},
    "train_method/dime": {"hard_budget"},
    "train_method/gdfs": {"hard_budget"},
}


def _experiment_files() -> list[Path]:
    return sorted(SCRIPTS_CONF_ROOT.glob("*/*/experiment/*.yaml"))


def _method_key(experiment_file: Path) -> str:
    script_group, method, _experiment_dir = experiment_file.relative_to(
        SCRIPTS_CONF_ROOT
    ).parts[:3]
    return f"{script_group}/{method}"


@pytest.mark.parametrize(
    "experiment_file",
    _experiment_files(),
    ids=[str(f.relative_to(REPO_ROOT)) for f in _experiment_files()],
)
def test_experiment_file_does_not_set_an_unknown_contract_field(
    experiment_file: Path,
) -> None:
    content = OmegaConf.load(experiment_file)
    assert OmegaConf.is_dict(content), (
        f"{experiment_file} must be a mapping of config keys"
    )

    violations = set(content.keys()) & CONTRACT_FIELDS
    allowed = KNOWN_VIOLATIONS.get(_method_key(experiment_file), set())

    assert violations <= allowed, (
        f"{experiment_file} sets contract field(s) "
        f"{violations - allowed} not in KNOWN_VIOLATIONS"
    )
