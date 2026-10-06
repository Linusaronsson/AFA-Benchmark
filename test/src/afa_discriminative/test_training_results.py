"""Library training returns objects; the caller owns bundle persistence."""

from pathlib import Path

import pytest
import torch
from torchrl.modules import MLP

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMEPretrainingConfig,
    DIMETabularArchitectureConfig,
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.components.methods.discriminative.dime.train.tabular import (
    train_tabular,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSPretrainingConfig,
    GDFSTabularArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.tabular import (
    pretrain_tabular as pretrain_gdfs,
)
from afabench.components.methods.discriminative.gdfs.train.tabular import (
    train_tabular as train_gdfs,
)
from afabench.components.methods.static.cae.config import (
    CAETabularArchitectureConfig,
    CAETrainingConfig,
)
from afabench.components.methods.static.cae.train.tabular import (
    train_tabular as train_cae,
)
from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.components.methods.static.common.static_methods import (
    StaticBaseMethod,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets.datasets import CubeDataset
from afabench.training.inputs import load_inputs


@pytest.mark.pipeline
@pytest.mark.parametrize("method", ["dime", "gdfs"])
def test_pretraining_returns_classifier_without_saving(
    tmp_path: Path, method: str
) -> None:
    train_path = tmp_path / "train.bundle"
    val_path = tmp_path / "val.bundle"
    save_bundle(CubeDataset(n_samples=128, seed=0), train_path, metadata={})
    save_bundle(CubeDataset(n_samples=32, seed=1), val_path, metadata={})
    config_type = (
        DIMEPretrainingConfig if method == "dime" else GDFSPretrainingConfig
    )
    architecture_type = (
        DIMETabularArchitectureConfig
        if method == "dime"
        else GDFSTabularArchitectureConfig
    )
    pretrain = pretrain_tabular if method == "dime" else pretrain_gdfs
    cfg = config_type(
        train_dataset_bundle_path=str(train_path),
        val_dataset_bundle_path=str(val_path),
        classifier_bundle_path="unused.bundle",
        save_path=str(tmp_path / "script_owned.bundle"),
        initializer=InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        batch_size=32,
        lr=0.001,
        nepochs=1,
        patience=1,
        min_masking_probability=0.0,
        max_masking_probability=0.9,
        architecture=architecture_type("ReLU", [8, 8], 0.0),
    )

    result = pretrain(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, GreedyAFAClassifier)
    assert not Path(cfg.save_path).exists()
    caller_path = tmp_path / "caller.bundle"
    save_bundle(result, caller_path, metadata={})
    restored, _ = load_bundle(caller_path, device=torch.device("cpu"))
    assert isinstance(restored, GreedyAFAClassifier)


@pytest.mark.pipeline
@pytest.mark.parametrize("method", ["dime", "gdfs"])
def test_training_returns_method_without_saving(
    tmp_path: Path, method: str
) -> None:
    config_type = (
        DIMETrainingConfig if method == "dime" else GDFSTrainingConfig
    )
    architecture_type = (
        DIMETabularArchitectureConfig
        if method == "dime"
        else GDFSTabularArchitectureConfig
    )
    train = train_tabular if method == "dime" else train_gdfs
    result_type = DIMEAFAMethod if method == "dime" else GDFSAFAMethod
    method_kwargs = (
        {"eps": 0.05, "eps_decay": 0.2, "eps_steps": 1}
        if method == "dime"
        else {}
    )
    dataset = CubeDataset(n_samples=128, seed=0)
    train_path = tmp_path / "train.bundle"
    val_path = tmp_path / "val.bundle"
    pretrained_path = tmp_path / "pretrained.bundle"
    save_bundle(dataset, train_path, metadata={})
    save_bundle(CubeDataset(n_samples=32, seed=1), val_path, metadata={})
    architecture = {
        "type": "mlp",
        "in_features": dataset.feature_shape.numel() * 2,
        "out_features": dataset.label_shape.numel(),
        "hidden_units": [8, 8],
        "activation": "ReLU",
        "dropout": 0.0,
    }
    pretrained = GreedyAFAClassifier(
        MLP(
            in_features=architecture["in_features"],
            out_features=architecture["out_features"],
            num_cells=[8, 8],
            dropout=0.0,
            activation_class=torch.nn.ReLU,
        ),
        architecture,
        torch.device("cpu"),
    )
    save_bundle(pretrained, pretrained_path, metadata={})
    cfg = config_type(
        train_dataset_bundle_path=str(train_path),
        val_dataset_bundle_path=str(val_path),
        classifier_bundle_path="unused.bundle",
        pretrained_model_bundle_path=str(pretrained_path),
        save_path=str(tmp_path / "script_owned.bundle"),
        initializer=InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        hard_budget=3,
        soft_budget_param=None,
        batch_size=32,
        lr=0.001,
        nepochs=1,
        patience=1,
        **method_kwargs,
        architecture=architecture_type("ReLU", [8, 8], 0.0),
    )

    result = train(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, result_type)
    assert not Path(cfg.save_path).exists()
    caller_path = tmp_path / "caller.bundle"
    save_bundle(result, caller_path, metadata={})
    restored, _ = load_bundle(caller_path, device=torch.device("cpu"))
    assert isinstance(restored, result_type)


@pytest.mark.pipeline
def test_cae_training_returns_method_without_saving(tmp_path: Path) -> None:
    train_path = tmp_path / "train.bundle"
    val_path = tmp_path / "val.bundle"
    save_bundle(CubeDataset(n_samples=128, seed=0), train_path, metadata={})
    save_bundle(CubeDataset(n_samples=32, seed=1), val_path, metadata={})
    cfg = CAETrainingConfig(
        train_dataset_bundle_path=str(train_path),
        val_dataset_bundle_path=str(val_path),
        classifier_bundle_path="unused.bundle",
        save_path=str(tmp_path / "script_owned.bundle"),
        initializer=InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        hard_budget=3,
        soft_budget_param=None,
        batch_size=32,
        architecture=CAETabularArchitectureConfig(
            StaticSelectorConfig(
                lr=0.001, nepochs=1, patience=1, num_cells=[8, 8]
            ),
            StaticClassifierConfig(lr=0.001, nepochs=1, num_cells=[8, 8]),
        ),
    )

    result = train_cae(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, StaticBaseMethod)
    assert not Path(cfg.save_path).exists()
    caller_path = tmp_path / "caller.bundle"
    save_bundle(result, caller_path, metadata={})
    restored, _ = load_bundle(caller_path, device=torch.device("cpu"))
    assert isinstance(restored, StaticBaseMethod)
