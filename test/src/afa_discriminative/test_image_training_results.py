"""Image training uses generated images and offline ImageNet weight loading."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.common.architecture import ArchitectureKind
from afabench.components.methods.discriminative.common import models
from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMEImageArchitectureConfig,
    DIMEPretrainingConfig,
    DIMETabularArchitectureConfig,
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.train import (
    train as train_dime,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSPretrainingConfig,
    GDFSTabularArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.image import (
    pretrain_image as pretrain_gdfs,
)
from afabench.components.methods.discriminative.gdfs.train import (
    train as train_gdfs,
)
from afabench.components.methods.static.cae.config import (
    CAEImageArchitectureConfig,
    CAETrainingConfig,
)
from afabench.components.methods.static.cae.train.image import (
    train_image as train_cae,
)
from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.components.methods.static.common.static_methods import (
    StaticBaseMethod,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.datasets import ImagenetteDataset
from afabench.fit.inputs import load_inputs
from afabench.testing.provenance import placeholder_provenance


@pytest.fixture
def image_contract_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    # Replace only the external weight-download boundary, not the backbone.
    weights = models.resnet18(pretrained=False).state_dict()
    monkeypatch.setattr(
        models, "load_state_dict_from_url", lambda *_args, **_kwargs: weights
    )
    root = tmp_path / "images"
    for class_index in range(2):
        directory = root / "generated" / "train" / str(class_index)
        directory.mkdir(parents=True)
        Image.new("RGB", (224, 224), color=(class_index * 200, 30, 40)).save(
            directory / "instance.png"
        )
    dataset = ImagenetteDataset(
        data_root=str(root),
        variant_dir="generated",
        load_subdirs=("train",),
        split_role="val",
    )
    path = tmp_path / "images.bundle"
    save_bundle(
        dataset, path, metadata={}, provenance=placeholder_provenance()
    )
    return {
        "train_dataset_bundle_path": str(path),
        "val_dataset_bundle_path": str(path),
        "classifier_bundle_path": "unused.bundle",
        "save_path": str(tmp_path / "script_owned.bundle"),
        "initializer": InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        "unmasker": UnmaskerConfig(
            "ImagePatchUnmasker",
            {"image_side_length": 224, "n_channels": 3, "patch_size": 16},
        ),
        "dataset_key": "imagenette",
        "device": "cpu",
        "seed": 0,
        "smoke_test": True,
    }


@pytest.mark.pipeline
@pytest.mark.parametrize("method", ["dime", "gdfs"])
def test_image_pretraining_returns_classifier_without_saving(
    image_contract_values: dict[str, object], method: str
) -> None:
    config_type = (
        DIMEPretrainingConfig if method == "dime" else GDFSPretrainingConfig
    )
    image_architecture_type = (
        DIMEImageArchitectureConfig
        if method == "dime"
        else GDFSImageArchitectureConfig
    )
    tabular_architecture_type = (
        DIMETabularArchitectureConfig
        if method == "dime"
        else GDFSTabularArchitectureConfig
    )
    pretrain = pretrain_image if method == "dime" else pretrain_gdfs
    cfg = config_type(
        **image_contract_values,
        batch_size=2,
        lr=0.001,
        nepochs=1,
        patience=1,
        min_masking_probability=0.0,
        max_masking_probability=0.9,
        architecture_kind=ArchitectureKind.IMAGE,
        tabular_architecture=tabular_architecture_type("ReLU", [8, 8], 0.0),
        image_architecture=image_architecture_type("resnet18", 224, 16),
    )

    result = pretrain(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, GreedyAFAClassifier)
    assert not Path(cfg.save_path).exists()


@pytest.mark.pipeline
@pytest.mark.parametrize("method", ["dime", "gdfs"])
@pytest.mark.parametrize("non_unit_costs", [False, True])
def test_image_training_returns_cost_aware_method_without_saving(
    image_contract_values: dict[str, object],
    tmp_path: Path,
    method: str,
    non_unit_costs: bool,
) -> None:
    config_type = (
        DIMETrainingConfig if method == "dime" else GDFSTrainingConfig
    )
    image_architecture_type = (
        DIMEImageArchitectureConfig
        if method == "dime"
        else GDFSImageArchitectureConfig
    )
    tabular_architecture_type = (
        DIMETabularArchitectureConfig
        if method == "dime"
        else GDFSTabularArchitectureConfig
    )
    train = train_dime if method == "dime" else train_gdfs
    result_type = DIMEAFAMethod if method == "dime" else GDFSAFAMethod
    method_kwargs = (
        {"eps": 0.05, "eps_decay": 0.2, "eps_steps": 1}
        if method == "dime"
        else {}
    )
    pretrain_cfg = DIMEPretrainingConfig(
        **image_contract_values,
        batch_size=2,
        lr=0.001,
        nepochs=1,
        patience=1,
        min_masking_probability=0.0,
        max_masking_probability=0.9,
        architecture_kind=ArchitectureKind.IMAGE,
        tabular_architecture=DIMETabularArchitectureConfig(
            "ReLU", [8, 8], 0.0
        ),
        image_architecture=DIMEImageArchitectureConfig("resnet18", 224, 16),
    )
    if non_unit_costs:
        dataset = load_inputs(pretrain_cfg).train_dataset()
        costs = torch.ones(dataset.feature_shape)
        # Selection 1 averages to 0.5 across its pixels and channels.
        costs[:, :8, 16:32] = 0.25
        costs[:, 8:16, 16:32] = 0.75
        dataset.feature_costs = costs
        save_bundle(
            dataset,
            Path(pretrain_cfg.train_dataset_bundle_path),
            metadata={},
            provenance=placeholder_provenance(),
        )
    pretrained = pretrain_image(pretrain_cfg, inputs=load_inputs(pretrain_cfg))
    pretrained_path = tmp_path / "pretrained.bundle"
    save_bundle(
        pretrained,
        pretrained_path,
        metadata={},
        provenance=placeholder_provenance(),
    )
    cfg = config_type(
        **image_contract_values,
        pretrained_model_bundle_path=str(pretrained_path),
        method_name="cae",
        hard_budget=1,
        soft_budget_param=None,
        batch_size=2,
        lr=0.001,
        min_lr=0.000001,
        nepochs=1,
        patience=1,
        **method_kwargs,
        tabular_architecture=tabular_architecture_type("ReLU", [8, 8], 0.0),
        image_architecture=image_architecture_type("resnet18", 224, 16),
    )

    result = train(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, result_type)
    assert not Path(cfg.save_path).exists()

    # Supply equal, positive scores to isolate the trained method's cost
    # ranking from learned scores, without depending on its architecture.
    scores = nn.Linear(3, 196)
    with torch.no_grad():
        scores.weight.zero_()
        scores.bias.fill_(1.0)
    result.policy_network = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), scores
    )
    action = result.act(
        masked_features=torch.zeros(2, 3, 224, 224),
        feature_mask=torch.zeros(2, 3, 224, 224, dtype=torch.bool),
        selection_mask=torch.zeros(2, 196, dtype=torch.bool),
        feature_shape=torch.Size((3, 224, 224)),
    )

    # Unit costs preserve the first-selection tie break; otherwise the
    # equally informative but cheaper second patch wins (action 2).
    expected_action = 2 if non_unit_costs else 1
    assert torch.equal(action, torch.full((2, 1), expected_action))


@pytest.mark.pipeline
def test_cae_image_training_returns_method_without_saving(
    image_contract_values: dict[str, object],
) -> None:
    cfg = CAETrainingConfig(
        **image_contract_values,
        method_name="cae",
        hard_budget=1,
        soft_budget_param=None,
        batch_size=2,
        selector=StaticSelectorConfig(
            lr=0.001, nepochs=1, patience=1, num_cells=[8, 8]
        ),
        classifier=StaticClassifierConfig(
            lr=0.001, nepochs=1, num_cells=[8, 8]
        ),
        architecture_kind=ArchitectureKind.IMAGE,
        image_architecture=CAEImageArchitectureConfig("resnet18", 224, 16),
    )

    result = train_cae(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, StaticBaseMethod)
    assert not Path(cfg.save_path).exists()
