"""L2M pretraining returns the pretrained model; the caller saves it."""

from pathlib import Path

import pytest
import torch

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.l2m.config import (
    L2MArchitectureConfig,
    L2MPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.methods.discriminative.l2m.pretrain import (
    pretrain_l2m,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets.datasets import CubeDataset
from afabench.fit.inputs import load_inputs
from afabench.testing.provenance import placeholder_provenance


def _save_dataset(dataset: CubeDataset, path: Path) -> str:
    save_bundle(
        dataset, path, metadata={}, provenance=placeholder_provenance()
    )
    return str(path)


def _config(
    tmp_path: Path, train_dataset: CubeDataset, feature_source: str
) -> L2MPretrainingConfig:
    return L2MPretrainingConfig(
        train_dataset_bundle_path=_save_dataset(
            train_dataset, tmp_path / "train.bundle"
        ),
        val_dataset_bundle_path=_save_dataset(
            CubeDataset(n_samples=32, seed=1), tmp_path / "val.bundle"
        ),
        classifier_bundle_path="unused.bundle",
        save_path=str(tmp_path / "script_owned.bundle"),
        initializer=InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        feature_source=feature_source,
        architecture=L2MArchitectureConfig(
            model_dim=8,
            embedding_depth=2,
            n_layers=1,
            n_heads=2,
            feedforward_dim=16,
        ),
        sequence_length=12,
        batch_size=2,
        n_steps=2,
        lr=1e-3,
        warmup_steps=1,
        checkpoint_interval=1,
        n_validation_tasks=2,
        missingness_cap=0.5,
    )


@pytest.mark.parametrize("feature_source", ["real", "synthetic"])
def test_pretraining_returns_model_without_saving(
    tmp_path: Path, feature_source: str
) -> None:
    cfg = _config(tmp_path, CubeDataset(n_samples=64, seed=0), feature_source)

    result = pretrain_l2m(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, L2MModel)
    assert not Path(cfg.save_path).exists()
    caller_path = tmp_path / "caller.bundle"
    save_bundle(
        result, caller_path, metadata={}, provenance=placeholder_provenance()
    )
    restored, _ = load_bundle(caller_path, device=torch.device("cpu"))
    assert isinstance(restored, L2MModel)
    features = torch.randn(5, result.n_features)
    mask = torch.rand(5, result.n_features) < 0.5
    labels = torch.eye(result.n_classes)[torch.tensor([0, 1, 2, 0, 0])]
    with torch.no_grad():
        expected = result(features, mask, labels, n_context=3)
        actual = restored(features, mask, labels, n_context=3)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_pretraining_never_reads_train_labels(tmp_path: Path) -> None:
    dataset = CubeDataset(n_samples=64, seed=0)
    relabelled = CubeDataset(n_samples=64, seed=0)
    relabelled.labels = relabelled.labels.roll(1, dims=0)
    models = []
    for name, train_dataset in [("a", dataset), ("b", relabelled)]:
        (tmp_path / name).mkdir()
        cfg = _config(tmp_path / name, train_dataset, "real")
        torch.manual_seed(0)
        models.append(pretrain_l2m(cfg, inputs=load_inputs(cfg)))

    for original, other in zip(
        models[0].parameters(), models[1].parameters(), strict=True
    ):
        assert torch.equal(original, other)
