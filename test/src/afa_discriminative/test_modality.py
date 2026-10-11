"""`prepare_modality` sets up tabular and image training without training."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchrl.modules import MLP

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.common import models
from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)
from afabench.components.methods.discriminative.common.modality import (
    prepare_modality,
)
from afabench.components.methods.discriminative.common.models import (
    build_resnet18_predictor,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
    MaskLayer2d,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSTabularArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.datasets.datasets import CubeDataset, ImagenetteDataset
from afabench.fit.inputs import load_inputs
from afabench.testing.provenance import placeholder_provenance

CUBE_ARCHITECTURE = {
    "type": "mlp",
    "in_features": 40,
    "out_features": 8,
    "hidden_units": [8, 8],
    "activation": "ReLU",
    "dropout": 0.0,
}
IMAGENETTE_ARCHITECTURE = {
    "type": "resnet18",
    "backbone": "resnet18",
    "image_size": 224,
    "patch_size": 16,
    "mask_width": 14,
    "d_out": 10,
}


def _save_dataset(dataset: object, path: Path) -> str:
    save_bundle(
        dataset, path, metadata={}, provenance=placeholder_provenance()
    )
    return str(path)


def _contract_values(
    tmp_path: Path, train_path: str, val_path: str
) -> dict[str, object]:
    return {
        "train_dataset_bundle_path": train_path,
        "val_dataset_bundle_path": val_path,
        "classifier_bundle_path": "unused.bundle",
        "pretrained_model_bundle_path": "unused.bundle",
        "save_path": str(tmp_path / "unused_save.bundle"),
        "initializer": InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        "dataset_key": "cube",
        "device": "cpu",
        "seed": 0,
        "method_name": "gdfs",
        "hard_budget": 3,
        "soft_budget_param": None,
        "lr": 0.001,
        "nepochs": 1,
        "patience": 1,
    }


@pytest.fixture
def tabular_cfg(tmp_path: Path) -> GDFSTrainingConfig:
    return GDFSTrainingConfig(
        **_contract_values(
            tmp_path,
            _save_dataset(
                CubeDataset(n_samples=64, seed=0), tmp_path / "train.bundle"
            ),
            _save_dataset(
                CubeDataset(n_samples=16, seed=1), tmp_path / "val.bundle"
            ),
        ),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
        batch_size=8,
        architecture=GDFSTabularArchitectureConfig("ReLU", [8, 8], 0.0),
    )


@pytest.fixture
def image_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> GDFSTrainingConfig:
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
    path = _save_dataset(
        ImagenetteDataset(
            data_root=str(root),
            variant_dir="generated",
            load_subdirs=("train",),
            split_role="val",
        ),
        tmp_path / "images.bundle",
    )
    return GDFSTrainingConfig(
        **_contract_values(tmp_path, path, path),
        unmasker=UnmaskerConfig(
            "ImagePatchUnmasker",
            {"image_side_length": 224, "n_channels": 3, "patch_size": 16},
        ),
        batch_size=2,
        min_lr=1e-6,
        architecture=GDFSImageArchitectureConfig("resnet18", 224, 16),
    )


def test_tabular_setup_batches_flat_features_with_class_indices(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        tabular_cfg,
        load_inputs(tabular_cfg),
        classifier_arch=CUBE_ARCHITECTURE,
    )

    assert (setup.d_in, setup.d_out, setup.n_selections) == (20, 8, 20)
    assert isinstance(setup.mask_layer, MaskLayer)
    features, labels = next(iter(setup.train_loader))
    assert features.shape == (8, 20)
    assert features.dtype == torch.float32
    assert labels.shape == (8,)
    assert labels.dtype == torch.long


def test_image_setup_selects_patches_of_the_pretrained_geometry(
    image_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        image_cfg,
        load_inputs(image_cfg),
        classifier_arch=IMAGENETTE_ARCHITECTURE,
    )

    assert (setup.d_out, setup.n_selections) == (10, 196)
    assert isinstance(setup.mask_layer, MaskLayer2d)
    assert setup.mask_layer.mask_width == 14
    features, _ = next(iter(setup.train_loader))
    assert features.shape == (2, 3, 224, 224)


def test_tabular_setup_weights_classes_and_charges_dataset_costs(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    inputs = load_inputs(tabular_cfg)

    setup = prepare_modality(
        tabular_cfg, inputs, classifier_arch=CUBE_ARCHITECTURE
    )

    _, labels = inputs.train_dataset().get_all_data()
    counts = torch.bincount(labels.argmax(dim=1), minlength=8).float()
    assert setup.class_weights is not None
    torch.testing.assert_close(
        setup.class_weights, (1 / counts) / (1 / counts).sum()
    )
    assert setup.feature_costs is not None
    torch.testing.assert_close(setup.feature_costs, torch.ones(20))
    assert setup.selection_costs is not None
    torch.testing.assert_close(setup.selection_costs, torch.ones(20))


def test_image_setup_trains_unweighted_on_unit_selection_costs(
    image_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        image_cfg,
        load_inputs(image_cfg),
        classifier_arch=IMAGENETTE_ARCHITECTURE,
    )

    assert setup.class_weights is None
    assert setup.feature_costs is None
    assert setup.selection_costs is None


def test_tabular_policy_network_scores_every_selection(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        tabular_cfg,
        load_inputs(tabular_cfg),
        classifier_arch=CUBE_ARCHITECTURE,
    )

    setup.check_policy_output(setup.policy_network())
    with pytest.raises(ValueError, match="19"):
        setup.check_policy_output(
            MLP(in_features=40, out_features=19, num_cells=[8])
        )


def test_tabular_policy_network_shares_the_predictors_first_linears(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        tabular_cfg,
        load_inputs(tabular_cfg),
        classifier_arch=CUBE_ARCHITECTURE,
    )
    predictor = MLP(in_features=40, out_features=8, num_cells=[8, 8])

    policy_network = setup.policy_network(share_with=predictor)

    predictor_linears = [
        m for m in predictor.modules() if isinstance(m, nn.Linear)
    ]
    policy_linears = [
        m for m in policy_network.modules() if isinstance(m, nn.Linear)
    ]
    assert policy_linears[:2] == predictor_linears[:2]
    setup.check_policy_output(policy_network)


def test_image_policy_network_scores_every_patch(
    image_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        image_cfg,
        load_inputs(image_cfg),
        classifier_arch=IMAGENETTE_ARCHITECTURE,
    )
    predictor = build_resnet18_predictor(IMAGENETTE_ARCHITECTURE)

    policy_network = setup.policy_network(share_with=predictor)

    assert policy_network.backbone is predictor.backbone
    setup.check_policy_output(policy_network)


def test_tabular_setup_falls_back_to_the_upstream_min_lr(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        tabular_cfg,
        load_inputs(tabular_cfg),
        classifier_arch=CUBE_ARCHITECTURE,
    )

    assert setup.min_lr(None, upstream_default=1e-5) == 1e-5
    assert setup.min_lr(1e-3, upstream_default=1e-5) == 1e-3


def test_image_setup_requires_a_configured_min_lr(
    image_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        image_cfg,
        load_inputs(image_cfg),
        classifier_arch=IMAGENETTE_ARCHITECTURE,
    )

    assert setup.min_lr(1e-8, upstream_default=1e-5) == 1e-8
    with pytest.raises(ValueError, match="min_lr"):
        setup.min_lr(None, upstream_default=1e-5)


def test_tabular_setup_records_the_mlp_pair(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        tabular_cfg,
        load_inputs(tabular_cfg),
        classifier_arch=CUBE_ARCHITECTURE,
    )

    assert setup.network_pair_spec() == NetworkPairSpec(
        modality="tabular",
        d_in=20,
        d_out=8,
        n_selections=20,
        policy_hidden_layers=[8, 8],
        predictor_hidden_layers=[8, 8],
        dropout=0.0,
    )


def test_image_setup_records_the_pretrained_geometry(
    image_cfg: GDFSTrainingConfig,
) -> None:
    setup = prepare_modality(
        image_cfg,
        load_inputs(image_cfg),
        classifier_arch=IMAGENETTE_ARCHITECTURE,
    )

    assert setup.network_pair_spec() == NetworkPairSpec(
        modality="image",
        d_out=10,
        backbone="resnet18",
        image_size=224,
        patch_size=16,
        mask_width=14,
    )


def test_unknown_pretrained_model_type_names_the_type(
    tabular_cfg: GDFSTrainingConfig,
) -> None:
    with pytest.raises(ValueError, match="'vit'"):
        prepare_modality(
            tabular_cfg,
            load_inputs(tabular_cfg),
            classifier_arch={**CUBE_ARCHITECTURE, "type": "vit"},
        )
