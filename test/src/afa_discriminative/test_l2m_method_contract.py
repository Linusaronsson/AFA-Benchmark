"""L2M's public method and model contracts with tiny untrained weights."""

from pathlib import Path

import pytest
import torch

from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
)
from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.testing.provenance import placeholder_provenance


def _make_model() -> L2MModel:
    torch.manual_seed(7)
    return L2MModel(
        3,
        2,
        model_dim=8,
        embedding_depth=2,
        n_layers=2,
        n_heads=2,
        feedforward_dim=16,
    )


def _make_method() -> L2MAFAMethod:
    return L2MAFAMethod(
        _make_model(),
        torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        torch.eye(2),
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
    )


def test_l2m_act_excludes_performed_selections_without_stop() -> None:
    method = _make_method()
    # Every selection mask over three features with a selection left.
    performed = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ]
    )
    features = torch.randn(
        len(performed), 3, generator=torch.Generator().manual_seed(0)
    )
    mask = torch.zeros_like(performed)

    actions = method.act(features, mask, selection_mask=performed).squeeze(-1)

    # Action k selects feature k - 1; action 0 would stop.
    assert torch.all((actions >= 1) & (actions <= 3))
    assert not performed.gather(1, (actions - 1).unsqueeze(-1)).any()
    # Where one selection is left, act must make it.
    assert torch.equal(actions[4:], torch.tensor([3, 2, 1]))


def test_l2m_predict_returns_classifier_logits_and_ignores_query_labels() -> (
    None
):
    method = _make_method()
    features = torch.tensor([[0.5, 0.0, -1.0], [0.0, 2.0, 0.0]])
    mask = features != 0

    prediction = method.predict(features, mask, label=torch.eye(2))

    assert method.has_builtin_classifier
    assert method.output_kind == "logits"
    torch.testing.assert_close(prediction, method.predict(features, mask))
    # The model's classifier logits for the queries after the context set.
    with torch.no_grad():
        expected, _ = method.model(
            torch.cat((method.context_features, features)),
            torch.cat((torch.ones(2, 3, dtype=torch.bool), mask)),
            torch.cat((method.context_labels, torch.eye(2))),
            n_context=2,
        )
    torch.testing.assert_close(prediction, expected)


def test_l2m_query_batching_does_not_change_actions_or_predictions() -> None:
    method = _make_method()
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [-4.0, 20.0, 0.0], [0.0, 0.0, 100.0]]
    )
    mask = features != 0

    alone_action = method.act(features[:1], mask[:1])
    alone_prediction = method.predict(features[:1], mask[:1])

    assert torch.equal(alone_action, method.act(features, mask)[:1])
    torch.testing.assert_close(
        alone_prediction, method.predict(features, mask)[:1]
    )


def test_l2m_rejects_non_direct_unmasker_configuration() -> None:
    with pytest.raises(ValueError, match=r"DirectUnmasker.*CubeNMUnmasker"):
        L2MAFAMethod(
            _make_model(),
            torch.zeros(2, 3),
            torch.eye(2),
            unmasker=UnmaskerConfig("CubeNMUnmasker", {}),
        )


def test_l2m_method_bundle_preserves_context_actions_and_logits(
    tmp_path: Path,
) -> None:
    method = _make_method()
    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
    mask = features != 0
    path = tmp_path / "method.bundle"
    save_bundle(method, path, {}, provenance=placeholder_provenance())

    restored, _ = load_bundle(path, device=torch.device("cpu"))

    assert isinstance(restored, L2MAFAMethod)
    assert restored.device == torch.device("cpu")
    assert torch.equal(restored.context_features, method.context_features)
    assert torch.equal(restored.context_labels, method.context_labels)
    assert torch.equal(
        restored.act(features, mask), method.act(features, mask)
    )
    assert torch.equal(
        restored.predict(features, mask), method.predict(features, mask)
    )


@pytest.mark.parametrize("explicit_selection_mask", [False, True])
def test_l2m_act_rejects_exhausted_selections(
    explicit_selection_mask: bool,
) -> None:
    method = _make_method()
    mask = torch.tensor([[True, True, True], [False, False, False]])
    with pytest.raises(ValueError, match=r"selection.*exhausted"):
        method.act(
            torch.zeros(2, 3),
            mask,
            selection_mask=mask if explicit_selection_mask else None,
        )


@pytest.mark.parametrize("shape", [(2, 2), (1, 3), (3,)])
def test_l2m_act_rejects_incompatible_selection_mask(
    shape: tuple[int, ...],
) -> None:
    method = _make_method()
    with pytest.raises(ValueError, match="selection_mask shape"):
        method.act(
            torch.zeros(2, 3),
            torch.zeros(2, 3, dtype=torch.bool),
            selection_mask=torch.zeros(shape, dtype=torch.bool),
        )


def test_l2m_model_batch_of_tasks_preserves_mask_gradients() -> None:
    model = _make_model()
    features = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0], [0.5, 2.0, -1.0]],
            [[3.0, 1.0, 2.0], [4.0, -1.0, 2.0], [-0.5, 1.0, 2.0]],
        ]
    )
    mask = torch.ones_like(features)
    mask[:, 2] = 0
    mask.requires_grad_()
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]] * 2)

    classifier, policy = model(features, mask, labels, n_context=2)
    assert classifier.shape == (2, 1, 2)
    assert policy.shape == (2, 1, 3)
    torch.nn.functional.cross_entropy(
        classifier.reshape(2, 2), torch.tensor([0, 1])
    ).backward()

    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()
    assert mask.grad[:, 2].abs().sum() > 0
    assert model.classifier_head.weight.grad is not None


def test_l2m_model_context_order_and_query_labels_do_not_leak() -> None:
    model = _make_model().eval()
    features = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0], [4.0, 1.0, 0.0]]
    )
    mask = torch.tensor([[True, True, True]] * 2 + [[True, False, False]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    classifier, policy = model(features, mask, labels, n_context=2)

    permuted = torch.tensor([1, 0, 2])
    reordered_labels = labels[permuted].clone()
    reordered_labels[-1] = torch.tensor([0.0, 1.0])
    reordered_features = features[permuted].clone()
    reordered_features[-1, 1:] = 500  # Unobserved values must be ignored.
    reordered_classifier, reordered_policy = model(
        reordered_features, mask[permuted], reordered_labels, n_context=2
    )

    torch.testing.assert_close(classifier, reordered_classifier)
    torch.testing.assert_close(policy, reordered_policy)


def test_l2m_model_bundle_rebuilds_architecture_and_both_heads(
    tmp_path: Path,
) -> None:
    model = L2MModel(
        3,
        4,
        model_dim=8,
        embedding_depth=1,
        n_layers=1,
        n_heads=2,
        feedforward_dim=12,
    ).eval()
    features = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    labels = torch.eye(4)[:2]
    mask = torch.ones_like(features, dtype=torch.bool)
    path = tmp_path / "pretrained.bundle"
    save_bundle(model, path, {}, provenance=placeholder_provenance())

    restored, _ = load_bundle(path, device=torch.device("cpu"))

    assert isinstance(restored, L2MModel)
    assert restored.architecture == model.architecture
    classifier, policy = restored(features, mask, labels, n_context=1)
    assert classifier.shape == (1, 4)
    assert policy.shape == (1, 3)
    expected_classifier, expected_policy = model(
        features, mask, labels, n_context=1
    )
    assert torch.equal(classifier, expected_classifier)
    assert torch.equal(policy, expected_policy)


@pytest.mark.parametrize("n_context", [-1, 0, 3, 4])
def test_l2m_model_requires_nonempty_context_and_queries(
    n_context: int,
) -> None:
    model = _make_model()
    with pytest.raises(ValueError, match=f"n_context={n_context}"):
        model(
            torch.zeros(3, 3),
            torch.ones(3, 3),
            torch.zeros(3, 2),
            n_context=n_context,
        )


@pytest.mark.parametrize(
    ("feature_shape", "mask_shape", "label_shape", "bad_input"),
    [
        ((3,), (3,), (2,), "features shape"),
        ((1, 2, 3, 3), (1, 2, 3, 3), (1, 2, 3, 2), "features shape"),
        ((3, 4), (3, 4), (3, 2), "features shape"),
        ((3, 3), (3,), (3, 2), "mask shape"),
        ((3, 3), (3, 3), (3, 1), "labels shape"),
    ],
)
def test_l2m_model_rejects_incompatible_task_shapes(
    feature_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    bad_input: str,
) -> None:
    with pytest.raises(ValueError, match=bad_input):
        _make_model()(
            torch.zeros(feature_shape),
            torch.ones(mask_shape),
            torch.zeros(label_shape),
            n_context=1,
        )


@pytest.mark.parametrize(
    ("feature_shape", "label_shape", "bad_input"),
    [
        ((0, 3), (0, 2), "context_features"),
        ((2, 4), (2, 2), "context_features"),
        ((2, 3), (1, 2), "context_labels"),
        ((2, 3), (2, 1), "context_labels"),
    ],
)
def test_l2m_method_rejects_invalid_context_set(
    feature_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    bad_input: str,
) -> None:
    with pytest.raises(ValueError, match=bad_input):
        L2MAFAMethod(
            _make_model(),
            torch.zeros(feature_shape),
            torch.zeros(label_shape),
            unmasker=UnmaskerConfig("DirectUnmasker", {}),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_features", 0),
        ("n_classes", 1),
        ("model_dim", 0),
        ("embedding_depth", 0),
        ("n_layers", 0),
        ("n_heads", 0),
        ("feedforward_dim", 0),
        ("model_dim", 7),
    ],
)
def test_l2m_model_rejects_invalid_architecture(
    parameter: str, value: int
) -> None:
    architecture = {
        "n_features": 3,
        "n_classes": 2,
        "model_dim": 8,
        "embedding_depth": 2,
        "n_layers": 1,
        "n_heads": 2,
        "feedforward_dim": 16,
    }
    architecture[parameter] = value
    with pytest.raises(ValueError, match=f"{parameter}={value}"):
        L2MModel(**architecture)


@pytest.mark.parametrize(
    ("feature_shape", "mask_shape", "bad_input"),
    [
        ((3,), (3,), "masked_features"),
        ((2, 4), (2, 4), "masked_features"),
        ((2, 3), (3,), "feature_mask"),
    ],
)
def test_l2m_predict_rejects_incompatible_query_shapes(
    feature_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    bad_input: str,
) -> None:
    with pytest.raises(ValueError, match=bad_input):
        _make_method().predict(
            torch.zeros(feature_shape),
            torch.zeros(mask_shape, dtype=torch.bool),
        )
