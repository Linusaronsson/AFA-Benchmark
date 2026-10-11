from dataclasses import (
    dataclass,
)
from pathlib import Path
from typing import Self, final, override

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from afabench.components.methods.common import (
    AFAMethodBase,
    load_classifier,
    on_method_device,
    save_classifier,
)
from afabench.core.types import (
    AFAAction,
    AFAClassifier,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
    SupportsForcedAcquisition,
)


@dataclass
@final
class RLAFAMethod(  # pyright: ignore[reportUnsafeMultipleInheritance]
    AFAMethodBase, SupportsForcedAcquisition
):
    """
    Implements the AFAMethod protocol for a TensorDictModule policy together with a classifier.

    The dataclass constructor sets the base's `_device` field itself, which
    is all the base's constructor does.
    """

    policy_tdmodule: TensorDictModuleBase | ProbabilisticActor
    afa_classifier: AFAClassifier
    _device: torch.device
    force_acquisition: bool = False

    def __post_init__(self):
        self.to(self._device)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    def get_td_from_masked_features(
        self,
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
        stop_action_mask = torch.full(
            selection_mask.shape[:-1] + (1,),
            (not self.force_acquisition),
            dtype=selection_mask.dtype,
            device=selection_mask.device,
        )
        action_mask = torch.cat(
            [
                stop_action_mask,
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
        assert selection_mask is not None, (
            "RLAFAMethod requires selection_mask"
        )

        batch_size = masked_features.shape[0]
        actions = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self._device
        )

        # Instances whose selection mask is fully performed have no
        # feature left to acquire, so stop without consulting the policy.
        active_mask = (~selection_mask).any(dim=-1)
        if not active_mask.any():
            return actions

        td = self.get_td_from_masked_features(
            masked_features[active_mask],
            feature_mask[active_mask],
            selection_mask[active_mask],
        )

        # Apply the agent's policy to the tensordict
        with (
            torch.no_grad(),
            set_exploration_type(ExplorationType.DETERMINISTIC),
        ):
            td = self.policy_tdmodule(td)

        # Get the action from the tensordict
        actions[active_mask] = td["action"].unsqueeze(-1)

        return actions

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        with torch.no_grad():
            return self.afa_classifier(
                masked_features, feature_mask, label, feature_shape
            )

    @override
    def save(self, path: Path) -> None:
        torch.save(self.policy_tdmodule, path / "policy_tdmodule.pt")
        save_classifier(self.afa_classifier, path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        policy_tdmodule = torch.load(
            path / "policy_tdmodule.pt",
            weights_only=False,
            map_location=device,
        )
        return cls(
            policy_tdmodule=policy_tdmodule,
            afa_classifier=load_classifier(path, device),
            _device=device,
        )
