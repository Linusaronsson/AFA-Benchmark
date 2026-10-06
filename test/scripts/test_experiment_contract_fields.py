"""Per-dataset experiments contain method hyperparameters, not contract inputs."""

from dataclasses import fields
from pathlib import Path

from omegaconf import OmegaConf

from afabench.training.contract import TrainingContract

CONF = Path(__file__).parents[2] / "extra/conf/scripts"
TABULAR_KEYS = {
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
}
# Existing exceptions only, removed as the remaining ports land (#52/#53).
LEGACY_FIELDS = {
    **{
        f"train_method/permutation/experiment/{key}.yaml": {"hard_budget"}
        for key in TABULAR_KEYS
    },
    **{
        f"pretrain_model/aaco/experiment/{key}.yaml": {"seed", "device"}
        for key in TABULAR_KEYS | {"cube_nm", "cube_nm_without_noise"}
    },
    **{
        f"train_method/aaco/experiment/{key}.yaml": {"seed", "device"}
        for key in (
            "actg",
            "bank_marketing",
            "ckd",
            "cube",
            "cube_nm",
            "diabetes",
            "fashion_mnist",
            "miniboone",
            "mnist",
            "physionet",
        )
    },
}


def test_experiment_files_do_not_set_training_contract_fields() -> None:
    contract_fields = {field.name for field in fields(TrainingContract)}
    violations = []
    for path in sorted(CONF.glob("*/*/experiment/*.yaml")):
        relative_path = path.relative_to(CONF).as_posix()
        experiment_fields = set(OmegaConf.load(path))
        forbidden = (experiment_fields & contract_fields) - LEGACY_FIELDS.get(
            relative_path, set()
        )
        if forbidden:
            violations.append(f"{relative_path}: {sorted(forbidden)}")
    assert not violations, "\n".join(violations)
