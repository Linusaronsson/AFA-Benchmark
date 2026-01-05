from typing import Any, Protocol

import torch
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModuleBase,
)
from torchrl.modules import ProbabilisticActor


class Agent(Protocol):
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        """Process one batch of data, updating network weights and replay buffer (if applicable). Return a wandb loggable dictionary containing info like losses and td errors (algorithm-dependent)."""
        ...

    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, Any]:
        """Return a wandb loggable dictionary when the tensordict contains full rollouts."""
        ...

    def get_cheap_info(self) -> dict[str, Any]:
        """Return a wandb loggable dictionary that is cheap to compute."""
        ...

    def get_expensive_info(self) -> dict[str, Any]:
        """Return a wandb loggable dictionary that is expensive to compute."""
        ...

    def get_policy(
        self,
    ) -> TensorDictModuleBase | ProbabilisticActor:
        """Alias for `get_exploratory_policy`."""
        ...

    def get_exploitative_policy(
        self,
    ) -> TensorDictModuleBase | ProbabilisticActor:
        """
        Return the agent's exploitative policy (if applicable).

        The caller is responsible for setting the ExplorationType.
        """
        ...

    def get_exploratory_policy(
        self,
    ) -> TensorDictModuleBase | ProbabilisticActor:
        """
        Return the agent's exploratory policy (if applicable).

        The caller is responsible for setting the ExplorationType.
        """
        ...

    def get_module_device(self) -> torch.device:
        """Retrieve the device that modules are currently placed on."""
        ...

    def set_module_device(self, device: torch.device) -> None:
        """Move agent modules to a specific device."""
        ...

    def get_replay_buffer_device(self) -> torch.device | None:
        """
        Retrieve the device that the replay buffer (if any) is currently placed on.

        Should return `None` if there is no replay buffer.
        """
        ...

    def set_replay_buffer_device(self, device: torch.device) -> None:
        """
        Move replay buffer (if any) to a specific device.

        For agents without a replay buffer, this should be a no-op.
        """
        ...
