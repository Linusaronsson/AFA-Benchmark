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

    def get_cheap_info(self) -> dict[str, Any]:
        """Return a wandb loggable dictionary containing info about the agent's state, without using too much compute."""
        ...

    def get_expensive_info(self) -> dict[str, Any]:
        """Return a wandb loggable dictionary containing info about the agent's state, possibly using very much compute."""
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

    # NOTE: save and load are not used currently. Instead, TensorDictModules are used directly

    # def save(self, path: Path) -> None:
    #     """Save the agent at the specified folder."""
    #     ...

    # @classmethod
    # def load(
    #     cls: type[Self],
    #     path: Path,
    #     module_device: torch.device,
    #     replay_buffer_device: torch.device,
    # ) -> Self:
    #     """Loads the agent from the specified folder, placing its modules and replay buffer (if any) on the specified devices."""
    #     ...
