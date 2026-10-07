from pathlib import Path

import pytest
import torch
from torch import nn

from afabench.components.classifiers import WrappedMaskedMLPClassifier
from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.components.initializers.config import InitializerConfig
from afabench.components.initializers.random_initializer import (
    RandomInitializer,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.components.unmaskers.direct_unmasker import DirectUnmasker
from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.datasets.datasets import CubeDataset
from afabench.training.contract import PretrainingContract, TrainingContract
from afabench.training.inputs import (
    UnavailableTrainingInputError,
    load_inputs,
)

N_FEATURES = 20
N_CLASSES = 8


def _training_contract(
    bundle_dir: Path, *, pretrained_model_bundle_path: str | None
) -> TrainingContract:
    return TrainingContract(
        train_dataset_bundle_path=str(bundle_dir / "train.bundle"),
        val_dataset_bundle_path=str(bundle_dir / "val.bundle"),
        classifier_bundle_path=str(bundle_dir / "classifier.bundle"),
        pretrained_model_bundle_path=pretrained_model_bundle_path,
        save_path=str(bundle_dir / "method.bundle"),
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        hard_budget=3,
        soft_budget_param=None,
        device="cpu",
        seed=0,
    )


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    save_bundle(
        CubeDataset(n_samples=10, seed=1),
        tmp_path / "train.bundle",
        metadata={},
    )
    save_bundle(
        CubeDataset(n_samples=6, seed=2),
        tmp_path / "val.bundle",
        metadata={},
    )
    save_bundle(
        WrappedMaskedMLPClassifier(
            MaskedMLPClassifier(
                n_features=N_FEATURES, n_classes=N_CLASSES, num_cells=(4,)
            ),
            device=torch.device("cpu"),
        ),
        tmp_path / "classifier.bundle",
        metadata={},
    )
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        tmp_path / "pretrained.bundle",
        metadata={},
    )
    return tmp_path


def test_datasets_are_loaded_from_their_bundles(bundle_dir: Path) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    train_features, _ = inputs.train_dataset().get_all_data()
    val_features, _ = inputs.val_dataset().get_all_data()

    assert train_features.shape == (10, N_FEATURES)
    assert val_features.shape == (6, N_FEATURES)


def test_initializer_and_unmasker_are_built_from_the_contract(
    bundle_dir: Path,
) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    assert isinstance(inputs.initializer(), RandomInitializer)
    assert isinstance(inputs.unmasker(), DirectUnmasker)


def test_classifier_is_loaded_on_the_contract_device(
    bundle_dir: Path,
) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    classifier = inputs.classifier(WrappedMaskedMLPClassifier)

    assert classifier.device == torch.device("cpu")
    assert classifier.module.n_classes == N_CLASSES


def test_inputs_are_loaded_once(bundle_dir: Path) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    assert inputs.train_dataset() is inputs.train_dataset()
    assert inputs.classifier(WrappedMaskedMLPClassifier) is inputs.classifier(
        WrappedMaskedMLPClassifier
    )


def test_inputs_are_not_loaded_before_they_are_requested(
    tmp_path: Path,
) -> None:
    inputs = load_inputs(
        _training_contract(tmp_path, pretrained_model_bundle_path=None)
    )

    with pytest.raises(FileNotFoundError):
        inputs.train_dataset()


def test_classifier_of_unexpected_type_is_rejected(bundle_dir: Path) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    with pytest.raises(
        TypeError,
        match=r"TorchModelBundle.*WrappedMaskedMLPClassifier",
    ):
        inputs.classifier(TorchModelBundle)


def test_pretrained_model_is_loaded_when_the_contract_provides_it(
    bundle_dir: Path,
) -> None:
    inputs = load_inputs(
        _training_contract(
            bundle_dir,
            pretrained_model_bundle_path=str(bundle_dir / "pretrained.bundle"),
        )
    )

    pretrained_model = inputs.pretrained_model(TorchModelBundle)

    assert isinstance(pretrained_model.model, nn.Linear)


def test_pretrained_model_is_unavailable_without_a_bundle_path(
    bundle_dir: Path,
) -> None:
    inputs = load_inputs(
        _training_contract(bundle_dir, pretrained_model_bundle_path=None)
    )

    with pytest.raises(
        UnavailableTrainingInputError, match="pretrained_model_bundle_path"
    ):
        inputs.pretrained_model(TorchModelBundle)


def test_pretraining_stage_provides_no_pretrained_model(
    bundle_dir: Path,
) -> None:
    contract = PretrainingContract(
        train_dataset_bundle_path=str(bundle_dir / "train.bundle"),
        val_dataset_bundle_path=str(bundle_dir / "val.bundle"),
        classifier_bundle_path=str(bundle_dir / "classifier.bundle"),
        save_path=str(bundle_dir / "model.bundle"),
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
    )
    inputs = load_inputs(contract)

    with pytest.raises(UnavailableTrainingInputError, match="pretraining"):
        inputs.pretrained_model(TorchModelBundle)
