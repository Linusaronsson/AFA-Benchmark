from pathlib import Path

import pytest
from torch import nn

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.generative.eddi.training import (
    build_eddi_afa_method,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.datasets.datasets import CubeDataset
from afabench.fit.contract import TrainingContract
from afabench.fit.inputs import load_inputs


def test_eddi_rejects_a_pretrained_model_that_is_not_a_partial_vae(
    tmp_path: Path,
) -> None:
    save_bundle(
        CubeDataset(n_samples=10, seed=1), tmp_path / "train.bundle", {}
    )
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)), tmp_path / "pretrained.bundle", {}
    )
    inputs = load_inputs(
        TrainingContract(
            train_dataset_bundle_path=str(tmp_path / "train.bundle"),
            val_dataset_bundle_path=str(tmp_path / "train.bundle"),
            classifier_bundle_path=str(tmp_path / "classifier.bundle"),
            pretrained_model_bundle_path=str(tmp_path / "pretrained.bundle"),
            save_path=str(tmp_path / "method.bundle"),
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
    )

    with pytest.raises(TypeError, match="Linear"):
        _ = build_eddi_afa_method(inputs, classifier_bundle_path=None)
