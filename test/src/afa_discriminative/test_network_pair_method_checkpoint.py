"""GDFS and DIME bundles: the model.pt format both methods write and read."""

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from afabench.components.methods.discriminative.common.afa_methods import (
    NetworkPairAFAMethod,
)
from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)
from afabench.components.methods.discriminative.common.models import (
    Classifier,
    ConvNet,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.testing.provenance import placeholder_provenance

CPU = torch.device("cpu")
D_IN = 6
D_OUT = 3
N_SELECTIONS = 4
TABULAR_SPEC = NetworkPairSpec(
    modality="tabular",
    d_in=D_IN,
    d_out=D_OUT,
    n_selections=N_SELECTIONS,
    policy_hidden_layers=[8],
    classifier_hidden_layers=[8, 8],
    dropout=0.0,
)
METHOD_CLASSES = [GDFSAFAMethod, DIMEAFAMethod]


def _tabular_method(
    method_class: type[NetworkPairAFAMethod],
) -> NetworkPairAFAMethod:
    torch.manual_seed(0)
    policy_network, classifier = TABULAR_SPEC.build_networks(CPU)
    return method_class(
        policy_network,
        classifier,
        TABULAR_SPEC,
        CPU,
        lambda_threshold=0.25,
        selection_costs=torch.tensor([1.0, 2.0, 1.0, 0.5]),
    )


def _episode_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    masked_features = torch.randn((5, D_IN))
    feature_mask = torch.rand((5, D_IN)) > 0.5
    selection_mask = torch.zeros((5, N_SELECTIONS), dtype=torch.bool)
    return masked_features, feature_mask, selection_mask


@pytest.mark.parametrize("method_class", METHOD_CLASSES)
def test_a_saved_method_loads_with_the_same_predictions_actions_and_costs(
    tmp_path: Path, method_class: type[NetworkPairAFAMethod]
) -> None:
    method = _tabular_method(method_class)
    masked_features, feature_mask, selection_mask = _episode_inputs()

    method.save(tmp_path)
    loaded = method_class.load(tmp_path, CPU)

    assert loaded.spec == TABULAR_SPEC
    assert loaded.cost_param == 0.25
    assert loaded.selection_costs is not None
    torch.testing.assert_close(
        loaded.selection_costs, torch.tensor([1.0, 2.0, 1.0, 0.5])
    )
    torch.testing.assert_close(
        loaded.predict(masked_features, feature_mask),
        method.predict(masked_features, feature_mask),
    )
    assert torch.equal(
        loaded.act(masked_features, feature_mask, selection_mask),
        method.act(masked_features, feature_mask, selection_mask),
    )


@pytest.mark.parametrize("method_class", METHOD_CLASSES)
def test_model_pt_uses_classifier_keys(
    tmp_path: Path, method_class: type[NetworkPairAFAMethod]
) -> None:
    method = _tabular_method(method_class)

    method.save(tmp_path)
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)

    policy_key = (
        "selector" if method_class is GDFSAFAMethod else "value_network"
    )
    assert set(checkpoint) == {
        f"{policy_key}_state_dict",
        "classifier_state_dict",
        "architecture",
        "lambda_threshold",
        "selection_costs",
    }
    assert f"{policy_key}_hidden_layers" in checkpoint["architecture"]
    assert "classifier_hidden_layers" in checkpoint["architecture"]


def test_missing_selection_fields_default_to_direct_unmasker_costs(
    tmp_path: Path,
) -> None:
    """Missing n_selections uses d_in; feature costs supply selection costs."""
    torch.manual_seed(0)
    legacy_spec = NetworkPairSpec(
        modality="tabular",
        d_in=D_IN,
        d_out=D_OUT,
        n_selections=D_IN,
        policy_hidden_layers=[8],
        classifier_hidden_layers=[8, 8],
        dropout=0.0,
    )
    selector, classifier = legacy_spec.build_networks(CPU)
    torch.save(
        {
            "selector_state_dict": selector.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "architecture": {
                "type": "mlp",
                "d_in": D_IN,
                "d_out": D_OUT,
                "selector_hidden_layers": [8],
                "classifier_hidden_layers": [8, 8],
                "dropout": 0.0,
                "model_type": "tabular",
            },
            "lambda_threshold": 0.5,
            "feature_costs": torch.ones(D_IN),
        },
        tmp_path / "model.pt",
    )

    loaded = GDFSAFAMethod.load(tmp_path, CPU)

    assert loaded.spec == legacy_spec
    assert loaded.cost_param == 0.5
    assert loaded.selection_costs is not None
    torch.testing.assert_close(loaded.selection_costs, torch.ones(D_IN))
    masked_features, feature_mask, _ = _episode_inputs()
    assert loaded.act(masked_features, feature_mask).shape == (5, 1)


@pytest.fixture
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.usefixtures("single_torch_thread")
@pytest.mark.parametrize("method_class", METHOD_CLASSES)
@pytest.mark.parametrize("lambda_threshold", [None, 0.25])
def test_image_bundle_preserves_trained_backbones_and_behaviour(
    tmp_path: Path,
    method_class: type[NetworkPairAFAMethod],
    lambda_threshold: float | None,
) -> None:
    spec = NetworkPairSpec(
        modality="image",
        d_out=3,
        backbone="resnet18",
        image_size=96,
        patch_size=16,
        mask_width=6,
    )
    with torch.random.fork_rng():
        torch.manual_seed(117)
        policy, classifier = spec.build_networks(CPU)
    assert isinstance(policy, ConvNet)
    assert isinstance(classifier, Classifier)
    if method_class is GDFSAFAMethod:
        policy.backbone = deepcopy(classifier.backbone)
        with torch.no_grad():
            next(policy.backbone.parameters()).fill_(0.01)
            next(classifier.backbone.parameters()).fill_(0.02)
    policy.eval()
    classifier.eval()
    method = method_class(
        policy, classifier, spec, CPU, lambda_threshold=lambda_threshold
    )
    rng = torch.Generator().manual_seed(117)
    features = torch.randn((2, 3, 96, 96), generator=rng)
    mask = torch.zeros_like(features, dtype=torch.bool)
    mask[:, :, :16, :16] = True
    features = features.masked_fill(~mask, 0)
    method.save(tmp_path)
    loaded = method_class.load(tmp_path, CPU)

    with torch.no_grad():
        torch.testing.assert_close(
            loaded.policy_network(features), policy(features)
        )
        torch.testing.assert_close(
            loaded.predict(features, mask), method.predict(features, mask)
        )
        assert torch.equal(
            loaded.act(features, mask), method.act(features, mask)
        )
    assert isinstance(loaded.policy_network, ConvNet)
    assert isinstance(loaded.classifier, Classifier)
    for original, restored in zip(
        policy.state_dict().values(),
        loaded.policy_network.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(restored, original)
    assert (loaded.policy_network.backbone is loaded.classifier.backbone) == (
        method_class is DIMEAFAMethod
    )


def test_a_method_without_soft_budget_parameter_never_stops_on_entropy(
    tmp_path: Path,
) -> None:
    policy_network, classifier = TABULAR_SPEC.build_networks(CPU)
    method = GDFSAFAMethod(policy_network, classifier, TABULAR_SPEC, CPU)

    method.save(tmp_path)
    loaded = GDFSAFAMethod.load(tmp_path, CPU)

    assert method.cost_param == loaded.cost_param == -float("inf")
    assert loaded.selection_costs is None


@pytest.mark.parametrize("method_class", METHOD_CLASSES)
def test_the_registry_rebuilds_the_method_from_its_bundle(
    tmp_path: Path, method_class: type[NetworkPairAFAMethod]
) -> None:
    method = _tabular_method(method_class)
    bundle_path = tmp_path / "method.bundle"

    save_bundle(
        method, bundle_path, metadata={}, provenance=placeholder_provenance()
    )
    loaded = load_bundle(bundle_path, device=CPU, expect=method_class).obj

    assert isinstance(loaded, method_class)
    assert loaded.spec == TABULAR_SPEC
