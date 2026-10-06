"""
Experiment files may not set training-contract fields.

Snakemake always overrides contract fields on the command line, so a
contract field set in an experiment YAML is dead in the pipeline and only
misleads a developer running the script by hand (see
`docs/training_contract_inventory.md`, issue #43). `PORTED_METHODS` lists
the methods whose experiment files have been checked clean by their
training-contract port; extend it as each further port lands
(`docs/adr/0001-training-contract-as-library.md`).
"""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.training.contract import PretrainingContract, TrainingContract

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_CONF = REPO_ROOT / "extra/conf/scripts"

PORTED_METHODS = {"random_dummy", "sequential_dummy"}

_STAGE_CONTRACT_FIELDS = {
    "train_method": {f.name for f in fields(TrainingContract)},
    "pretrain_model": {f.name for f in fields(PretrainingContract)},
}


def _experiment_files() -> list[tuple[str, Path, frozenset[str]]]:
    cases = []
    for stage, contract_fields in _STAGE_CONTRACT_FIELDS.items():
        stage_dir = SCRIPTS_CONF / stage
        if not stage_dir.is_dir():
            continue
        for method_dir in sorted(stage_dir.iterdir()):
            if method_dir.name not in PORTED_METHODS:
                continue
            experiment_dir = method_dir / "experiment"
            if not experiment_dir.is_dir():
                continue
            cases.extend(
                (
                    f"{stage}/{method_dir.name}/experiment/"
                    f"{experiment_file.name}",
                    experiment_file,
                    frozenset(contract_fields),
                )
                for experiment_file in sorted(experiment_dir.glob("*.yaml"))
            )
    return cases


@pytest.mark.parametrize(
    ("case_id", "experiment_file", "contract_fields"),
    [
        pytest.param(case_id, experiment_file, contract_fields, id=case_id)
        for case_id, experiment_file, contract_fields in _experiment_files()
    ],
)
def test_experiment_file_does_not_set_a_contract_field(
    case_id: str,  # noqa: ARG001
    experiment_file: Path,
    contract_fields: frozenset[str],
) -> None:
    content = OmegaConf.load(experiment_file)
    keys = {key for key in content if isinstance(key, str)}

    violations = keys & contract_fields

    assert violations == set(), (
        f"{experiment_file} sets contract field(s) {violations}"
    )
