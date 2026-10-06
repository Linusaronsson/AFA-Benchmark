"""
Experiment files may not set training-contract fields.

Snakemake always overrides contract fields on the command line, so a
contract field set in an experiment YAML is dead in the pipeline and only
misleads a developer running the script by hand (see
`docs/training_contract_inventory.md`, issue #43). This test covers every
method under `extra/conf/scripts/`. `NOT_YET_PORTED` lists the methods whose
experiment files are known to still set contract fields, pending their
training-contract port (see the open "Port ... to the training contract
helpers" issues); their cases are `xfail(strict=True)` so a forgotten
removal from the set turns into a failure once a port actually fixes the
files (`docs/adr/0001-training-contract-as-library.md`).
"""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.training.contract import PretrainingContract, TrainingContract

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_CONF = REPO_ROOT / "extra/conf/scripts"

_STAGE_CONTRACT_FIELDS = {
    "train_method": frozenset(f.name for f in fields(TrainingContract)),
    "pretrain_model": frozenset(f.name for f in fields(PretrainingContract)),
}

NOT_YET_PORTED: dict[str, frozenset[str]] = {
    "train_method": frozenset({"cae", "dime", "gdfs"}),
}


def _experiment_files() -> list[tuple[Path, frozenset[str], bool]]:
    cases = []
    for stage, contract_fields in _STAGE_CONTRACT_FIELDS.items():
        stage_dir = SCRIPTS_CONF / stage
        if not stage_dir.is_dir():
            msg = f"missing scripts conf stage dir: {stage_dir}"
            raise FileNotFoundError(msg)
        not_yet_ported = NOT_YET_PORTED.get(stage, frozenset())
        for method_dir in sorted(stage_dir.iterdir()):
            experiment_dir = method_dir / "experiment"
            if not experiment_dir.is_dir():
                continue
            is_known_violator = method_dir.name in not_yet_ported
            cases.extend(
                (experiment_file, contract_fields, is_known_violator)
                for experiment_file in sorted(experiment_dir.glob("*.yaml"))
            )
    return cases


def _case_id(experiment_file: Path) -> str:
    return "/".join(experiment_file.parts[-4:])


_CASES = _experiment_files()
assert _CASES, (
    f"no experiment files found under {SCRIPTS_CONF}; "
    "check that the stage/method layout did not change"
)


@pytest.mark.parametrize(
    ("experiment_file", "contract_fields"),
    [
        pytest.param(
            experiment_file,
            contract_fields,
            id=_case_id(experiment_file),
            marks=(
                [
                    pytest.mark.xfail(
                        reason=(
                            f"{experiment_file.parts[-3]} not yet ported "
                            "to the training contract"
                        ),
                        strict=True,
                    )
                ]
                if is_known_violator
                else []
            ),
        )
        for experiment_file, contract_fields, is_known_violator in _CASES
    ],
)
def test_experiment_file_does_not_set_a_contract_field(
    experiment_file: Path, contract_fields: frozenset[str]
) -> None:
    content = OmegaConf.load(experiment_file)
    keys = {key for key in content if isinstance(key, str)}

    violations = keys & contract_fields

    assert violations == set(), (
        f"{experiment_file} sets contract field(s) {violations}"
    )
