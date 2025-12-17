from dataclasses import (
    dataclass,
)
from pathlib import Path
from typing import Self, cast, final, override

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from afabench.common.custom_types import (
    AFAClassifier,
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_td_from_masked_features(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask,
) -> TensorDict:
    """
    Create a TensorDict suitable as input to AFA RL agents.

    The keys are:
    - "action_mask"
    - "masked_features"
    - "feature_mask"
    """
    # The action mask is almost the same as the negated selection mask but with one extra element (the stop action)
    action_mask = torch.cat(
        [
            torch.ones(
                selection_mask.shape[:-1] + (1,),
                dtype=feature_mask.dtype,
                device=feature_mask.device,
            ),
            ~selection_mask,
        ],
        dim=-1,
    )

    td = TensorDict(
        {
            "allowed_action_mask": action_mask,
            "masked_features": masked_features,
            "feature_mask": feature_mask,
        },
        batch_size=masked_features.shape[0],
        device=masked_features.device,
    )

    return td


@dataclass
@final
class RLAFAMethod(AFAMethod):
    """Implements the AFAMethod protocol for a TensorDictModule policy together with a classifier."""

    policy_tdmodule: TensorDictModuleBase | ProbabilisticActor
    afa_classifier: AFAClassifier
    _device: torch.device = torch.device("cpu")  # noqa: RUF009

    def __post_init__(self):
        # Move policy and classifier to the specified device
        self.policy_tdmodule = self.policy_tdmodule.to(self._device)
        self.afa_classifier = self.afa_classifier.to(self._device)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFASelection:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        assert selection_mask is not None, (
            "RLAFAMethod requires selection_mask"
        )
        td = get_td_from_masked_features(
            masked_features, feature_mask, selection_mask
        )

        # Apply the agent's policy to the tensordict
        with (
            torch.no_grad(),
            set_exploration_type(ExplorationType.DETERMINISTIC),
        ):
            td = self.policy_tdmodule(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            probs = self.afa_classifier(
                masked_features, feature_mask, label, feature_shape
            )
        return probs.to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.policy_tdmodule, path / "policy_tdmodule.pt")
        self.afa_classifier.save(path / "classifier.pt")
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        policy_tdmodule = torch.load(
            path / "policy_tdmodule.pt",
            weights_only=False,
            map_location=device,
        )

        with (path / "classifier_class_name.txt").open() as f:
            classifier_class_name = f.read()
        from afabench.common.registry import get_class

        afa_classifier = get_class(classifier_class_name).load(
            path / "classifier.pt", device=device
        )
        afa_classifier = cast("AFAClassifier", cast("object", afa_classifier))

        return cls(
            policy_tdmodule=policy_tdmodule,
            afa_classifier=afa_classifier,
            _device=device,
        )

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.policy_tdmodule = self.policy_tdmodule.to(self._device)
        self.afa_classifier = self.afa_classifier.to(self._device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
