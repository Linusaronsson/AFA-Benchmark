"""Dataset hyperparameters must not override the pipeline training contract."""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.training.contract import TrainingContract

SCRIPT_CONFIGS = Path(__file__).parents[2] / "extra/conf/scripts"
CONTRACT_FIELDS = {field.name for field in fields(TrainingContract)}
# Existing hard-budget overrides in methods awaiting their own #43 ports.
# Each port removes its entries; new paths or other fields are not exempt.
LEGACY_HARD_BUDGET_PATHS = {
    "train_method/cae/experiment/actg.yaml",
    "train_method/cae/experiment/bank_marketing.yaml",
    "train_method/cae/experiment/ckd.yaml",
    "train_method/cae/experiment/cube.yaml",
    "train_method/cae/experiment/cube_nonuniform_costs.yaml",
    "train_method/cae/experiment/cube_without_noise.yaml",
    "train_method/cae/experiment/diabetes.yaml",
    "train_method/cae/experiment/fashion_mnist.yaml",
    "train_method/cae/experiment/miniboone.yaml",
    "train_method/cae/experiment/mnist.yaml",
    "train_method/cae/experiment/physionet.yaml",
    "train_method/cae/experiment/synthetic_mnist.yaml",
    "train_method/cae/experiment/synthetic_mnist_without_noise.yaml",
    "train_method/dime/experiment/actg.yaml",
    "train_method/dime/experiment/bank_marketing.yaml",
    "train_method/dime/experiment/ckd.yaml",
    "train_method/dime/experiment/cube.yaml",
    "train_method/dime/experiment/cube_nm.yaml",
    "train_method/dime/experiment/cube_nm_without_noise.yaml",
    "train_method/dime/experiment/cube_nonuniform_costs.yaml",
    "train_method/dime/experiment/cube_without_noise.yaml",
    "train_method/dime/experiment/diabetes.yaml",
    "train_method/dime/experiment/fashion_mnist.yaml",
    "train_method/dime/experiment/imagenette.yaml",
    "train_method/dime/experiment/miniboone.yaml",
    "train_method/dime/experiment/mnist.yaml",
    "train_method/dime/experiment/physionet.yaml",
    "train_method/dime/experiment/synthetic_mnist.yaml",
    "train_method/dime/experiment/synthetic_mnist_without_noise.yaml",
    "train_method/gdfs/experiment/actg.yaml",
    "train_method/gdfs/experiment/bank_marketing.yaml",
    "train_method/gdfs/experiment/ckd.yaml",
    "train_method/gdfs/experiment/cube.yaml",
    "train_method/gdfs/experiment/cube_nm.yaml",
    "train_method/gdfs/experiment/cube_nm_without_noise.yaml",
    "train_method/gdfs/experiment/cube_nonuniform_costs.yaml",
    "train_method/gdfs/experiment/cube_without_noise.yaml",
    "train_method/gdfs/experiment/diabetes.yaml",
    "train_method/gdfs/experiment/fashion_mnist.yaml",
    "train_method/gdfs/experiment/imagenette.yaml",
    "train_method/gdfs/experiment/miniboone.yaml",
    "train_method/gdfs/experiment/mnist.yaml",
    "train_method/gdfs/experiment/physionet.yaml",
    "train_method/gdfs/experiment/synthetic_mnist.yaml",
    "train_method/gdfs/experiment/synthetic_mnist_without_noise.yaml",
    "train_method/permutation/experiment/actg.yaml",
    "train_method/permutation/experiment/bank_marketing.yaml",
    "train_method/permutation/experiment/ckd.yaml",
    "train_method/permutation/experiment/cube.yaml",
    "train_method/permutation/experiment/cube_nonuniform_costs.yaml",
    "train_method/permutation/experiment/cube_without_noise.yaml",
    "train_method/permutation/experiment/diabetes.yaml",
    "train_method/permutation/experiment/fashion_mnist.yaml",
    "train_method/permutation/experiment/miniboone.yaml",
    "train_method/permutation/experiment/mnist.yaml",
    "train_method/permutation/experiment/physionet.yaml",
    "train_method/permutation/experiment/synthetic_mnist.yaml",
    "train_method/permutation/experiment/synthetic_mnist_without_noise.yaml",
}


@pytest.mark.parametrize(
    "path",
    sorted(SCRIPT_CONFIGS.glob("*/*/experiment/*.yaml")),
    ids=lambda path: str(path.relative_to(SCRIPT_CONFIGS)),
)
def test_experiment_does_not_set_training_contract_fields(path: Path) -> None:
    config = OmegaConf.load(path)
    allowed = (
        {"hard_budget"}
        if str(path.relative_to(SCRIPT_CONFIGS)) in LEGACY_HARD_BUDGET_PATHS
        else set()
    )
    assert not (set(config) & CONTRACT_FIELDS) - allowed
