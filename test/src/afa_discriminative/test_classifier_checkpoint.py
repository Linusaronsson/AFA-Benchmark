"""Pretrained classifiers use the same vocabulary as method bundles."""

from pathlib import Path

import torch

from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
    build_mlp,
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.testing.provenance import placeholder_provenance


def test_classifier_bundle_round_trips_with_classifier_state_dict(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    architecture = {
        "type": "mlp",
        "in_features": 12,
        "out_features": 3,
        "hidden_units": [8],
        "activation": "ReLU",
        "dropout": 0.0,
    }
    classifier = GreedyAFAClassifier(
        classifier=build_mlp(architecture),
        architecture=architecture,
        device=device,
    )
    inputs = torch.ones((2, 12))
    expected = classifier.classifier(inputs)
    bundle_path = tmp_path / "classifier.bundle"

    save_bundle(
        classifier,
        bundle_path,
        metadata={},
        provenance=placeholder_provenance(),
    )
    loaded = load_bundle(
        bundle_path, device=device, expect=GreedyAFAClassifier
    ).obj

    assert loaded.architecture == architecture
    assert not loaded.classifier.training
    torch.testing.assert_close(loaded.classifier(inputs), expected)

    checkpoint_path = tmp_path / "checkpoint"
    loaded.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path / "model.pt", weights_only=False)
    assert set(checkpoint) == {"classifier_state_dict", "architecture"}
