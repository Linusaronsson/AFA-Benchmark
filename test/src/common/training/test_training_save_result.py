from dataclasses import dataclass
from pathlib import Path

from torch import nn

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.training.contract import PretrainingContract, TrainingContract
from afabench.training.run import save_result


@dataclass(frozen=True)
class _MethodTrainConfig(TrainingContract):
    learning_rate: float


@dataclass(frozen=True)
class _PretrainHyperparameters:
    n_epochs: int


def test_save_result_writes_a_loadable_bundle_to_save_path(
    tmp_path: Path,
) -> None:
    config = _MethodTrainConfig(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        pretrained_model_bundle_path="pretrained.bundle",
        save_path=str(tmp_path / "method.bundle"),
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        hard_budget=3,
        soft_budget_param=0.1,
        device="cpu",
        seed=7,
        learning_rate=0.5,
    )

    save_result(
        TorchModelBundle(nn.Linear(2, 2)),
        config,
        config,
        stage="training",
    )

    loaded, manifest = load_bundle(tmp_path / "method.bundle")
    assert isinstance(loaded, TorchModelBundle)
    assert manifest["metadata"] == {
        "stage": "training",
        "contract": {
            "train_dataset_bundle_path": "train.bundle",
            "val_dataset_bundle_path": "val.bundle",
            "classifier_bundle_path": "classifier.bundle",
            "pretrained_model_bundle_path": "pretrained.bundle",
            "save_path": str(tmp_path / "method.bundle"),
            "initializer": {
                "class_name": "RandomInitializer",
                "kwargs": {"num_initial_features": 0},
            },
            "unmasker": {"class_name": "DirectUnmasker", "kwargs": {}},
            "dataset_key": "cube",
            "hard_budget": 3,
            "soft_budget_param": 0.1,
            "device": "cpu",
            "seed": 7,
            "use_wandb": False,
            "smoke_test": False,
        },
        "method_config": {"learning_rate": 0.5},
    }


def test_save_result_records_a_separate_method_config(
    tmp_path: Path,
) -> None:
    contract = PretrainingContract(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        save_path=str(tmp_path / "model.bundle"),
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=1,
        smoke_test=True,
    )

    save_result(
        TorchModelBundle(nn.Linear(2, 2)),
        contract,
        _PretrainHyperparameters(n_epochs=4),
        stage="pretraining",
    )

    _, manifest = load_bundle(tmp_path / "model.bundle")
    metadata = manifest["metadata"]
    assert metadata["stage"] == "pretraining"
    assert set(metadata["contract"]) == {
        "train_dataset_bundle_path",
        "val_dataset_bundle_path",
        "classifier_bundle_path",
        "save_path",
        "initializer",
        "unmasker",
        "dataset_key",
        "device",
        "seed",
        "use_wandb",
        "smoke_test",
    }
    assert metadata["contract"]["smoke_test"] is True
    assert metadata["method_config"] == {"n_epochs": 4}
