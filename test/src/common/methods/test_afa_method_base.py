"""AFAMethodBase: the device and persistence behaviour every method shares."""

from pathlib import Path
from typing import Self, override

import pytest
import torch
from torch import nn

from afabench.components.classifiers import UniformDummyAFAClassifier
from afabench.components.methods.common import (
    AFAMethodBase,
    load_classifier,
    on_method_device,
    save_classifier,
)
from afabench.core.types import (
    AFAAction,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class _RecordingMethod(AFAMethodBase):
    """Holds one of each attribute kind `to` moves, and records the device its inputs arrive on."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        self.scale: nn.Linear = nn.Linear(3, 3)
        self.offset: torch.Tensor = torch.zeros(3)
        self.classifier: UniformDummyAFAClassifier = UniformDummyAFAClassifier(
            n_classes=2
        )
        self.input_devices: list[torch.device] = []

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @override
    @on_method_device
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        del feature_mask, selection_mask, label, feature_shape
        self.input_devices.append(masked_features.device)
        return torch.zeros(
            (masked_features.shape[0], 1), dtype=torch.long, device=self.device
        )

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        self.input_devices.append(masked_features.device)
        return self.classifier(
            masked_features, feature_mask, label, feature_shape
        )

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        raise NotImplementedError


def test_base_implements_the_afa_method_protocol() -> None:
    assert isinstance(_RecordingMethod(torch.device("cpu")), AFAMethod)


def test_to_moves_tensor_module_and_classifier_attributes(
    device_str: str,
) -> None:
    device = torch.device(device_str)
    method = _RecordingMethod(torch.device("cpu"))

    moved = method.to(device)

    assert moved is method
    assert method.device == device
    assert method.offset.device.type == device.type
    assert next(method.scale.parameters()).device.type == device.type
    assert method.classifier.device == device


def test_act_receives_its_inputs_on_the_method_device() -> None:
    # The meta device holds shape-only tensors, so it stands in for a second
    # device on a host without one. Nothing can be copied out of it, so this
    # probe answers on the caller's device itself.
    class _Probe(_RecordingMethod):
        @override
        @on_method_device
        def act(
            self,
            masked_features: MaskedFeatures,
            feature_mask: FeatureMask,
            selection_mask: SelectionMask | None = None,
            label: Label | None = None,
            feature_shape: torch.Size | None = None,
        ) -> AFAAction:
            del feature_mask, selection_mask, label, feature_shape
            self.input_devices.append(masked_features.device)
            return torch.zeros((masked_features.shape[0], 1), dtype=torch.long)

    method = _Probe(torch.device("cpu")).to(torch.device("meta"))

    method.act(torch.zeros((2, 3)), torch.zeros((2, 3), dtype=torch.bool))

    assert [device.type for device in method.input_devices] == ["meta"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second device"
)
def test_act_and_predict_answer_on_the_callers_device() -> None:
    method = _RecordingMethod(torch.device("cuda"))
    masked_features = torch.zeros((2, 3))
    feature_mask = torch.zeros((2, 3), dtype=torch.bool)

    action = method.act(masked_features, feature_mask)
    prediction = method.predict(masked_features, feature_mask)

    assert action.device.type == "cpu"
    assert prediction.device.type == "cpu"
    assert [device.type for device in method.input_devices] == [
        "cuda",
        "cuda",
    ]


def test_method_without_soft_budget_parameter_or_generator_accepts_the_evaluators_calls() -> (
    None
):
    method = _RecordingMethod(torch.device("cpu"))

    method.set_cost_param(0.5)
    method.set_seed(7)
    method.set_seed(None)

    assert method.cost_param is None


def test_has_builtin_classifier_must_be_declared() -> None:
    class _Undeclared(_RecordingMethod):
        # Put the base's abstract property back in place of the declaration
        has_builtin_classifier = AFAMethodBase.has_builtin_classifier

    with pytest.raises(TypeError, match="has_builtin_classifier"):
        _Undeclared(torch.device("cpu"))


def test_save_classifier_writes_the_layout_load_classifier_reads(
    tmp_path: Path,
) -> None:
    save_classifier(UniformDummyAFAClassifier(n_classes=3), tmp_path)

    assert (tmp_path / "classifier_class_name.txt").read_text() == (
        "UniformDummyAFAClassifier"
    )
    loaded = load_classifier(tmp_path, torch.device("cpu"))
    assert isinstance(loaded, UniformDummyAFAClassifier)
    assert loaded.n_classes == 3


def test_load_classifier_reads_the_layout_saved_before_the_base(
    tmp_path: Path,
) -> None:
    # The two files as the dummy and RL methods wrote them by hand
    (tmp_path / "classifier_class_name.txt").write_text(
        "UniformDummyAFAClassifier"
    )
    torch.save(4, tmp_path / "classifier.pt")

    loaded = load_classifier(tmp_path, torch.device("cpu"))

    assert isinstance(loaded, UniformDummyAFAClassifier)
    assert loaded.n_classes == 4
