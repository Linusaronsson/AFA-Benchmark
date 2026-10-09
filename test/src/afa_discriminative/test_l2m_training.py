"""L2M training returns the method with its context set; the caller saves."""

from pathlib import Path

import pytest
import torch

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
)
from afabench.components.methods.discriminative.l2m.config import (
    L2MTrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.methods.discriminative.l2m.task_sampler import (
    FeatureSource,
)
from afabench.components.methods.discriminative.l2m.train import train_l2m
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets.datasets import CubeDataset
from afabench.fit.inputs import load_inputs
from afabench.testing.provenance import placeholder_provenance

N_FEATURES = 20
N_CLASSES = 8


def _save(obj: object, path: Path) -> str:
    save_bundle(obj, path, metadata={}, provenance=placeholder_provenance())
    return str(path)


def _pretrained_model() -> L2MModel:
    torch.manual_seed(3)
    return L2MModel(
        N_FEATURES,
        N_CLASSES,
        model_dim=8,
        embedding_depth=2,
        n_layers=1,
        n_heads=2,
        feedforward_dim=16,
    )


def _config(
    tmp_path: Path,
    *,
    train_dataset: CubeDataset | None = None,
    val_dataset: CubeDataset | None = None,
    seed: int = 0,
    context_set_size: int = 8,
    unmasker: UnmaskerConfig | None = None,
    hard_budget: int | None = 3,
) -> L2MTrainingConfig:
    return L2MTrainingConfig(
        train_dataset_bundle_path=_save(
            train_dataset or CubeDataset(n_samples=64, seed=0),
            tmp_path / "train.bundle",
        ),
        val_dataset_bundle_path=_save(
            val_dataset or CubeDataset(n_samples=32, seed=1),
            tmp_path / "val.bundle",
        ),
        classifier_bundle_path="unused.bundle",
        save_path=str(tmp_path / "script_owned.bundle"),
        initializer=InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        unmasker=unmasker or UnmaskerConfig("DirectUnmasker", {}),
        dataset_key="cube",
        device="cpu",
        seed=seed,
        method_name="l2m_real_feature_prior",
        pretrained_model_bundle_path=_save(
            _pretrained_model(), tmp_path / "pretrained.bundle"
        ),
        hard_budget=hard_budget,
        soft_budget_param=None,
        feature_source=FeatureSource.real,
        sequence_length=12,
        batch_size=2,
        n_steps=2,
        policy_lr=1e-3,
        backbone_lr=1e-4,
        temperature=0.1,
        checkpoint_interval=1,
        n_validation_tasks=2,
        missingness_cap=0.5,
        context_set_size=context_set_size,
    )


def test_training_returns_method_without_saving_and_bundle_keeps_context(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)

    result = train_l2m(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, L2MAFAMethod)
    assert not Path(cfg.save_path).exists()
    assert result.context_features.shape == (8, N_FEATURES)
    assert result.context_labels.shape == (8, N_CLASSES)
    restored, _ = load_bundle(
        Path(_save(result, tmp_path / "caller.bundle")),
        device=torch.device("cpu"),
    )
    assert isinstance(restored, L2MAFAMethod)
    assert torch.equal(restored.context_features, result.context_features)
    assert torch.equal(restored.context_labels, result.context_labels)
    features = torch.randn(5, N_FEATURES)
    mask = torch.rand(5, N_FEATURES) < 0.5
    assert torch.equal(
        restored.act(features, mask), result.act(features, mask)
    )
    assert torch.equal(
        restored.predict(features, mask), result.predict(features, mask)
    )


def _are_instances_of(
    instances: torch.Tensor, dataset_instances: torch.Tensor
) -> bool:
    return all(
        any(
            torch.equal(instance, candidate) for candidate in dataset_instances
        )
        for instance in instances
    )


@pytest.mark.optional
def test_context_set_is_validation_instances_drawn_with_the_contract_seed(
    tmp_path: Path,
) -> None:
    val_dataset = CubeDataset(n_samples=32, seed=1)
    val_features, val_labels = val_dataset.get_all_data()
    context_sets = {}
    for name, seed in [("a", 0), ("b", 0), ("c", 1)]:
        (tmp_path / name).mkdir()
        cfg = _config(tmp_path / name, val_dataset=val_dataset, seed=seed)
        method = train_l2m(cfg, inputs=load_inputs(cfg))
        context_sets[name] = (method.context_features, method.context_labels)
        assert _are_instances_of(
            torch.cat(context_sets[name], dim=1),
            torch.cat((val_features, val_labels), dim=1),
        )

    assert torch.equal(context_sets["a"][0], context_sets["b"][0])
    assert torch.equal(context_sets["a"][1], context_sets["b"][1])
    assert not torch.equal(context_sets["a"][0], context_sets["c"][0])


def test_training_rejects_validation_split_smaller_than_context_set(
    tmp_path: Path,
) -> None:
    cfg = _config(
        tmp_path,
        val_dataset=CubeDataset(n_samples=7, seed=1),
        context_set_size=8,
    )
    with pytest.raises(ValueError, match=r"7 instances.*context_set_size=8"):
        train_l2m(cfg, inputs=load_inputs(cfg))


@pytest.mark.optional
def test_training_never_reads_train_labels(tmp_path: Path) -> None:
    dataset = CubeDataset(n_samples=64, seed=0)
    relabelled = CubeDataset(n_samples=64, seed=0)
    relabelled.labels = relabelled.labels.roll(1, dims=0)
    methods = []
    for name, train_dataset in [("a", dataset), ("b", relabelled)]:
        (tmp_path / name).mkdir()
        cfg = _config(tmp_path / name, train_dataset=train_dataset)
        methods.append(train_l2m(cfg, inputs=load_inputs(cfg)))

    for original, other in zip(
        methods[0].model.parameters(),
        methods[1].model.parameters(),
        strict=True,
    ):
        assert torch.equal(original, other)


def test_training_rejects_non_direct_unmasker_before_training(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path, unmasker=UnmaskerConfig("CubeNMUnmasker", {}))
    with pytest.raises(ValueError, match=r"DirectUnmasker.*CubeNMUnmasker"):
        train_l2m(cfg, inputs=load_inputs(cfg))


@pytest.mark.parametrize("hard_budget", [None, 0, N_FEATURES])
def test_training_requires_a_hard_budget_below_the_selection_count(
    tmp_path: Path, hard_budget: int | None
) -> None:
    cfg = _config(tmp_path, hard_budget=hard_budget)
    with pytest.raises(ValueError, match=f"hard_budget={hard_budget}"):
        train_l2m(cfg, inputs=load_inputs(cfg))
