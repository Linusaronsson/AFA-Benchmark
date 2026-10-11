"""NetworkPairSpec: the architecture record of a GDFS or DIME checkpoint."""

import pytest
import torch

from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)

CPU = torch.device("cpu")


def test_tabular_spec_round_trips_through_the_checkpoint_dict() -> None:
    spec = NetworkPairSpec(
        modality="tabular",
        d_in=20,
        d_out=8,
        n_selections=18,
        policy_hidden_layers=[64, 32],
        classifier_hidden_layers=[128, 128],
        dropout=0.3,
    )

    arch = spec.to_dict("selector")

    assert arch == {
        "type": "mlp",
        "d_in": 20,
        "d_out": 8,
        "n_selections": 18,
        "selector_hidden_layers": [64, 32],
        "classifier_hidden_layers": [128, 128],
        "dropout": 0.3,
        "model_type": "tabular",
    }
    assert NetworkPairSpec.from_dict(arch, "selector") == spec


def test_image_spec_round_trips_through_the_checkpoint_dict() -> None:
    spec = NetworkPairSpec(
        modality="image",
        d_out=10,
        backbone="resnet18",
        image_size=224,
        patch_size=16,
        mask_width=14,
    )

    arch = spec.to_dict("value_network")

    assert arch == {
        "type": "resnet18",
        "backbone": "resnet18",
        "image_size": 224,
        "patch_size": 16,
        "mask_width": 14,
        "d_out": 10,
        "model_type": "image",
    }
    assert NetworkPairSpec.from_dict(arch, "value_network") == spec


def test_tabular_dict_without_n_selections_falls_back_to_d_in() -> None:
    """Checkpoints written before grouped selections had no n_selections."""
    arch = {
        "type": "mlp",
        "d_in": 20,
        "d_out": 8,
        "value_network_hidden_layers": [128, 128],
        "classifier_hidden_layers": [128, 128],
        "dropout": 0.3,
    }

    spec = NetworkPairSpec.from_dict(arch, "value_network")

    assert spec.n_selections == 20


def test_unknown_checkpoint_type_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="vit"):
        NetworkPairSpec.from_dict({"type": "vit", "d_out": 10}, "selector")


@pytest.mark.parametrize(
    ("missing", "values"),
    [
        (
            "n_selections",
            {
                "modality": "tabular",
                "d_in": 4,
                "d_out": 2,
                "policy_hidden_layers": [8],
                "classifier_hidden_layers": [8],
                "dropout": 0.0,
            },
        ),
        (
            "patch_size",
            {
                "modality": "image",
                "d_out": 2,
                "backbone": "resnet18",
                "image_size": 32,
                "mask_width": 2,
            },
        ),
    ],
)
def test_spec_missing_a_field_its_modality_needs_is_rejected(
    missing: str, values: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=missing):
        NetworkPairSpec(**values)  # pyright: ignore[reportArgumentType]


def test_tabular_networks_take_features_with_mask_and_answer_per_selection_and_class() -> (
    None
):
    spec = NetworkPairSpec(
        modality="tabular",
        d_in=6,
        d_out=3,
        n_selections=4,
        policy_hidden_layers=[8],
        classifier_hidden_layers=[8, 8],
        dropout=0.0,
    )

    policy_network, classifier = spec.build_networks(CPU)

    features_with_mask = torch.zeros((2, 2 * 6))
    assert policy_network(features_with_mask).shape == (2, 4)
    assert classifier(features_with_mask).shape == (2, 3)
