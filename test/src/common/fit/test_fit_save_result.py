from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from afabench.components.classifiers import WrappedMaskedMLPClassifier
from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import (
    bundle_provenance,
    load_bundle,
    read_manifest,
    save_bundle,
)
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.provenance import (
    DatasetIdentityMismatchError,
    ProvenanceInput,
)
from afabench.datasets.datasets import CubeDataset
from afabench.fit.contract import PretrainingContract, TrainingContract
from afabench.fit.run import save_result
from afabench.testing.provenance import placeholder_provenance


@dataclass(frozen=True)
class _MethodTrainConfig(TrainingContract):
    learning_rate: float


@dataclass(frozen=True)
class _PretrainHyperparameters:
    n_epochs: int


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """Input bundles of one dataset realization, with provenance records."""
    save_bundle(
        CubeDataset(n_samples=10, seed=1),
        tmp_path / "train.bundle",
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube", dataset_realization_index=2, split="train"
        ),
    )
    save_bundle(
        CubeDataset(n_samples=6, seed=2),
        tmp_path / "val.bundle",
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube", dataset_realization_index=2, split="val"
        ),
    )
    save_bundle(
        WrappedMaskedMLPClassifier(
            MaskedMLPClassifier(n_features=20, n_classes=8, num_cells=(4,)),
            device=torch.device("cpu"),
        ),
        tmp_path / "classifier.bundle",
        metadata={},
        provenance=placeholder_provenance("classifier_training"),
    )
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        tmp_path / "pretrained.bundle",
        metadata={},
        provenance=placeholder_provenance("pretraining"),
    )
    return tmp_path


def _training_config(inputs: Path, **overrides: object) -> _MethodTrainConfig:
    values: dict[str, object] = {
        "train_dataset_bundle_path": str(inputs / "train.bundle"),
        "val_dataset_bundle_path": str(inputs / "val.bundle"),
        "classifier_bundle_path": str(inputs / "classifier.bundle"),
        "pretrained_model_bundle_path": str(inputs / "pretrained.bundle"),
        "save_path": str(inputs / "method.bundle"),
        "method_name": "my_method",
        "initializer": InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        "unmasker": UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        "dataset_key": "cube",
        "hard_budget": 3,
        "soft_budget_param": 0.1,
        "device": "cpu",
        "seed": 7,
        "learning_rate": 0.5,
    }
    values.update(overrides)
    return _MethodTrainConfig(**values)  # pyright: ignore[reportArgumentType]


def test_save_result_writes_a_loadable_bundle_to_save_path(
    inputs: Path,
) -> None:
    config = _training_config(inputs)

    save_result(TorchModelBundle(nn.Linear(2, 2)), config, config)

    loaded, manifest = load_bundle(inputs / "method.bundle")
    assert isinstance(loaded, TorchModelBundle)
    assert manifest["metadata"] == {
        "stage": "training",
        "contract": {
            "train_dataset_bundle_path": str(inputs / "train.bundle"),
            "val_dataset_bundle_path": str(inputs / "val.bundle"),
            "classifier_bundle_path": str(inputs / "classifier.bundle"),
            "pretrained_model_bundle_path": str(inputs / "pretrained.bundle"),
            "save_path": str(inputs / "method.bundle"),
            "method_name": "my_method",
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


def test_save_result_records_training_provenance(inputs: Path) -> None:
    config = _training_config(inputs, seed=11, smoke_test=True)

    save_result(TorchModelBundle(nn.Linear(2, 2)), config, config)

    record = bundle_provenance(inputs / "method.bundle")
    assert record is not None
    assert record.stage == "training"
    assert record.seed == 11
    assert record.smoke_test is True
    assert record.method_name == "my_method"
    assert (record.dataset_key, record.dataset_realization_index) == (
        "cube",
        2,
    )
    assert record.split is None
    assert record.resolved_config["learning_rate"] == 0.5
    assert record.resolved_config["seed"] == 11
    assert record.compute.device == "cpu"
    assert record.inputs == [
        ProvenanceInput(
            role="train_dataset",
            path=str(inputs / "train.bundle"),
            class_name="CubeDataset",
            content_hash=read_manifest(inputs / "train.bundle")[
                "content_hash"
            ],
        ),
        ProvenanceInput(
            role="val_dataset",
            path=str(inputs / "val.bundle"),
            class_name="CubeDataset",
            content_hash=read_manifest(inputs / "val.bundle")["content_hash"],
        ),
        ProvenanceInput(
            role="classifier",
            path=str(inputs / "classifier.bundle"),
            class_name="WrappedMaskedMLPClassifier",
            content_hash=read_manifest(inputs / "classifier.bundle")[
                "content_hash"
            ],
        ),
        ProvenanceInput(
            role="pretrained_model",
            path=str(inputs / "pretrained.bundle"),
            class_name="TorchModelBundle",
            content_hash=read_manifest(inputs / "pretrained.bundle")[
                "content_hash"
            ],
        ),
    ]


def test_save_result_without_a_pretrained_model_records_three_inputs(
    inputs: Path,
) -> None:
    config = _training_config(inputs, pretrained_model_bundle_path=None)

    save_result(TorchModelBundle(nn.Linear(2, 2)), config, config)

    record = bundle_provenance(inputs / "method.bundle")
    assert record is not None
    assert [entry.role for entry in record.inputs] == [
        "train_dataset",
        "val_dataset",
        "classifier",
    ]


def test_save_result_rejects_datasets_of_different_realizations(
    inputs: Path,
) -> None:
    save_bundle(
        CubeDataset(n_samples=6, seed=3),
        inputs / "other_val.bundle",
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube", dataset_realization_index=3, split="val"
        ),
    )
    config = _training_config(
        inputs, val_dataset_bundle_path=str(inputs / "other_val.bundle")
    )

    with pytest.raises(DatasetIdentityMismatchError, match=r"2.*3"):
        save_result(TorchModelBundle(nn.Linear(2, 2)), config, config)


def test_save_result_records_a_separate_method_config(inputs: Path) -> None:
    contract = PretrainingContract(
        train_dataset_bundle_path=str(inputs / "train.bundle"),
        val_dataset_bundle_path=str(inputs / "val.bundle"),
        classifier_bundle_path=str(inputs / "classifier.bundle"),
        save_path=str(inputs / "model.bundle"),
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
    )

    _, manifest = load_bundle(inputs / "model.bundle")
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
    record = bundle_provenance(inputs / "model.bundle")
    assert record is not None
    assert record.stage == "pretraining"
    assert record.method_name is None
    assert record.resolved_config["n_epochs"] == 4
    assert record.resolved_config["seed"] == 1
    assert [entry.role for entry in record.inputs] == [
        "train_dataset",
        "val_dataset",
        "classifier",
    ]
