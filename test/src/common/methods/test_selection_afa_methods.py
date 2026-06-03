from pathlib import Path

import torch

from afabench.components.classifiers import UniformDummyAFAClassifier
from afabench.components.methods.dummy import (
    RandomWithClassifierAFAMethod,
    SequentialWithClassifierAFAMethod,
)


def test_random_selection_method_selects_unobserved_feature() -> None:
    classifier = UniformDummyAFAClassifier(n_classes=3)
    method = RandomWithClassifierAFAMethod(classifier)
    feature_mask = torch.tensor([[True, False, True]])

    selection = method.act(
        masked_features=torch.zeros((1, 3)),
        feature_mask=feature_mask,
    )

    assert selection.item() == 2


def test_random_selection_method_stops_when_all_features_observed() -> None:
    classifier = UniformDummyAFAClassifier(n_classes=3)
    method = RandomWithClassifierAFAMethod(classifier)
    feature_mask = torch.ones((2, 3), dtype=torch.bool)

    selection = method.act(
        masked_features=torch.zeros((2, 3)),
        feature_mask=feature_mask,
    )

    assert torch.equal(selection, torch.zeros((2, 1), dtype=torch.long))


def test_sequential_selection_method_selects_first_unobserved_feature() -> (
    None
):
    classifier = UniformDummyAFAClassifier(n_classes=3)
    method = SequentialWithClassifierAFAMethod(classifier)
    feature_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )

    selection = method.act(
        masked_features=torch.zeros((2, 3)),
        feature_mask=feature_mask,
    )

    assert torch.equal(selection, torch.tensor([[2], [1]]))


def test_selection_methods_use_classifier_for_prediction() -> None:
    classifier = UniformDummyAFAClassifier(n_classes=3)
    method = SequentialWithClassifierAFAMethod(classifier)

    prediction = method.predict(
        masked_features=torch.zeros((2, 3)),
        feature_mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    assert torch.equal(prediction, torch.ones((2, 3)))


def test_selection_method_roundtrip(tmp_path: Path) -> None:
    classifier = UniformDummyAFAClassifier(n_classes=3)
    method = RandomWithClassifierAFAMethod(classifier)

    method.save(tmp_path)
    loaded = RandomWithClassifierAFAMethod.load(tmp_path, torch.device("cpu"))

    prediction = loaded.predict(
        masked_features=torch.zeros((2, 3)),
        feature_mask=torch.zeros((2, 3), dtype=torch.bool),
    )
    assert torch.equal(prediction, torch.ones((2, 3)))
