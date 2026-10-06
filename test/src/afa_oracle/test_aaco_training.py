from pathlib import Path

import torch

from afabench.components.classifiers import WrappedMaskedMLPClassifier
from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.oracle.aaco.afa_methods import AACOAFAMethod
from afabench.components.methods.oracle.aaco.config import (
    AACOConfig,
    AACOTrainConfig,
)
from afabench.components.methods.oracle.aaco.train import run
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.datasets.datasets import CubeDataset


def test_aaco_training_returns_a_method_without_saving(tmp_path: Path) -> None:
    dataset = CubeDataset(n_samples=16, seed=0)
    dataset_path = tmp_path / "train.bundle"
    classifier_path = tmp_path / "classifier.bundle"
    save_path = tmp_path / "method.bundle"
    save_bundle(dataset, dataset_path, metadata={})
    save_bundle(
        WrappedMaskedMLPClassifier(
            MaskedMLPClassifier(
                n_features=dataset.feature_shape.numel(),
                n_classes=dataset.label_shape.numel(),
                num_cells=(8,),
            ),
            device=torch.device("cpu"),
        ),
        classifier_path,
        metadata={},
    )
    cfg = AACOTrainConfig(
        train_dataset_bundle_path=str(dataset_path),
        val_dataset_bundle_path=str(tmp_path / "unused_val.bundle"),
        classifier_bundle_path=str(classifier_path),
        save_path=str(save_path),
        initializer=InitializerConfig(
            class_name="RandomInitializer", kwargs={"num_initial_features": 0}
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        hard_budget=3,
        soft_budget_param=None,
        aco=AACOConfig(),
    )

    method = run(cfg)

    assert isinstance(method, AACOAFAMethod)
    assert not save_path.exists()
    features, _ = dataset.get_all_data()
    prediction = method.predict(
        masked_features=features[:1],
        feature_mask=torch.ones_like(features[:1], dtype=torch.bool),
        feature_shape=dataset.feature_shape,
    )
    assert prediction.shape == (1, dataset.label_shape.numel())
