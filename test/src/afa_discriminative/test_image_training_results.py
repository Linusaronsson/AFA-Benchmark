"""Image training uses generated images and offline ImageNet weight loading."""

from pathlib import Path

import pytest
from PIL import Image

from afabench.components.initializers.config import InitializerConfig
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
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSPretrainingConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.image import (
    pretrain_image as pretrain_gdfs,
)
from afabench.components.methods.discriminative.gdfs.train.image import (
    train_image as train_gdfs,
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
from afabench.datasets.datasets import ImagenetteDataset
from afabench.training.inputs import load_inputs


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
    save_bundle(dataset, path, metadata={})
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
    architecture_type = (
        DIMEImageArchitectureConfig
        if method == "dime"
        else GDFSImageArchitectureConfig
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
        architecture=architecture_type("resnet18", 224, 16),
    )

    result = pretrain(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, GreedyAFAClassifier)
    assert not Path(cfg.save_path).exists()


@pytest.mark.pipeline
@pytest.mark.parametrize("method", ["dime", "gdfs"])
def test_image_training_returns_method_without_saving(
    image_contract_values: dict[str, object], tmp_path: Path, method: str
) -> None:
    config_type = (
        DIMETrainingConfig if method == "dime" else GDFSTrainingConfig
    )
    architecture_type = (
        DIMEImageArchitectureConfig
        if method == "dime"
        else GDFSImageArchitectureConfig
    )
    train = train_image if method == "dime" else train_gdfs
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
        architecture=DIMEImageArchitectureConfig("resnet18", 224, 16),
    )
    pretrained = pretrain_image(pretrain_cfg, inputs=load_inputs(pretrain_cfg))
    pretrained_path = tmp_path / "pretrained.bundle"
    save_bundle(pretrained, pretrained_path, metadata={})
    cfg = config_type(
        **image_contract_values,
        pretrained_model_bundle_path=str(pretrained_path),
        hard_budget=1,
        soft_budget_param=None,
        batch_size=2,
        lr=0.001,
        min_lr=0.000001,
        nepochs=1,
        patience=1,
        **method_kwargs,
        architecture=architecture_type("resnet18", 224, 16),
    )

    result = train(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, result_type)
    assert not Path(cfg.save_path).exists()


@pytest.mark.pipeline
def test_cae_image_training_returns_method_without_saving(
    image_contract_values: dict[str, object],
) -> None:
    cfg = CAETrainingConfig(
        **image_contract_values,
        hard_budget=1,
        soft_budget_param=None,
        batch_size=2,
        architecture=CAEImageArchitectureConfig(
            StaticSelectorConfig(
                lr=0.001, nepochs=1, patience=1, num_cells=[8, 8]
            ),
            StaticClassifierConfig(lr=0.001, nepochs=1, num_cells=[8, 8]),
            "resnet18",
            224,
            16,
        ),
    )

    result = train_cae(cfg, inputs=load_inputs(cfg))

    assert isinstance(result, StaticBaseMethod)
    assert not Path(cfg.save_path).exists()
