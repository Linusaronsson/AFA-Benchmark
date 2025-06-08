from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar, override

import torch
import yaml
from tensordict import TensorDictBase
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import Tensor, nn, optim
from torchrl.data import (
    LazyTensorStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import ClipPPOLoss, DQNLoss, SoftUpdate, ValueEstimators
from torchrl.objectives.value import GAE

T = TypeVar("T", bound="Agent")


def field_with_metadata(
    field_type: str, default=MISSING, default_factory=MISSING, init: bool = True
):
    """Create a dataclass field with metadata."""
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    metadata = {"type": field_type}
    # If default is provided, prefer it over default_factory
    if default is not MISSING:
        return field(default=default, init=init, metadata=metadata)
    elif default_factory is not MISSING:
        return field(default_factory=default_factory, init=init, metadata=metadata)
    else:
        return field(init=init, metadata=metadata)


def serializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as serializable (saved to params.yml)."""
    return field_with_metadata("serializable", default, default_factory, init)


def weights(default=MISSING, default_factory=MISSING, init: bool = False):
    """Mark a field as containing model weights (saved to model.pt)."""
    return field_with_metadata("weights", default, default_factory, init)


def unserializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as unserializable (saved to extra.pt)."""
    return field_with_metadata("unserializable", default, default_factory, init)


class Agent(ABC):
    # Subclasses will be dataclasses
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @abstractmethod
    def process_batch(self, td: TensorDictBase) -> dict[str, float]: ...

    def get_train_info(self) -> dict[str, Any]:
        return {}

    def get_eval_info(self) -> dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def policy(self) -> TensorDictModuleBase: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @device.setter
    @abstractmethod
    def device(self, device: torch.device) -> None:
        """Move agent networks to a specific device."""
        ...

    def to(self, device: torch.device) -> Self:
        self.device = device
        return self

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save the class name
        with open(path / "class_name.txt", "w") as f:
            f.write(self.__class__.__name__)

        # Save weights
        weights_dict = {
            field.name: getattr(self, field.name).state_dict()
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "weights"
        }
        torch.save(weights_dict, path / "model.pt")

        # Save unserializable fields
        unserializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "unserializable"
        }
        torch.save(unserializable_dict, path / "extra.pt")

        # Save serializable fields
        serializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "serializable"
        }
        with open(path / "params.yml", "w") as f:
            yaml.dump(serializable_dict, f)

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        # Load the class name
        with open(path / "class_name.txt") as f:
            class_name = f.read().strip()

        # Find the subclass, even if it's several layers down
        def find_subclass(base_class: type, name: str) -> type | None:
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == name:
                    return subclass
                found = find_subclass(subclass, name)
                if found:
                    return found
            return None

        subclass = find_subclass(cls, class_name)
        if subclass is None:
            raise ValueError(f"Unknown subclass: {class_name}")

        # Load serializable fields
        with open(path / "params.yml") as f:
            serializable_dict = yaml.safe_load(f)

        # Load unserializable fields
        unserializable_dict = torch.load(path / "extra.pt", weights_only=False, map_location=torch.device("cpu"))

        # Merge all fields
        full_dict = {**serializable_dict, **unserializable_dict}

        # Construct agent
        agent = subclass(**full_dict)

        # Load weights
        weights_dict = torch.load(path / "model.pt", map_location=torch.device("cpu"))
        for _field in agent.__dataclass_fields__.values():
            if _field.metadata.get("type") == "weights":
                getattr(agent, _field.name).load_state_dict(weights_dict[_field.name])

        return agent


@dataclass(kw_only=True, eq=False, order=False)
class DQNAgent(Agent, ABC):
    """Deep Q-Network (DQN) agent."""

    action_spec: TensorSpec = unserializable()

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    action_mask_key: str | None = serializable(None)

    # DQN parameters
    gamma: float = serializable(default=0.99)
    loss_function: str = serializable(default="l2")
    delay_value: bool = serializable(default=True)
    double_dqn: bool = serializable(default=False)

    # epsilon greedy parameters
    eps_annealing_num_batches: int = serializable(default=10000)
    eps_init: float = serializable(default=1.0)
    eps_end: float = serializable(default=0.1)

    # target network update rate
    update_tau: float = serializable(default=0.005)

    # Optimizer parameters
    lr: float = serializable(default=1e-3)
    max_grad_norm: float = serializable(default=1)

    # Replay buffer parameters
    replay_buffer_size: int = serializable(default=1000)
    replay_buffer_batch_size: int = serializable(default=100)
    num_optim: int = serializable(default=10)
    replay_buffer_alpha: float = serializable(default=0.6)
    replay_buffer_beta_init: float = serializable(default=0.4)
    replay_buffer_beta_end: float = serializable(default=1.0)
    replay_buffer_beta_annealing_num_batches: int = serializable(default=10000)
    init_random_frames: int = serializable(default=0)
    replay_buffer_device: torch.device = unserializable(
        default_factory=lambda: torch.device("cpu")
    )

    # Set in constructor
    action_value_module: TensorDictModule = weights(init=False)
    greedy_module: QValueModule = field(init=False)
    greedy_policy_module: TensorDictModule = field(init=False)
    egreedy_module: EGreedyModule = field(init=False)
    egreedy_policy_module: TensorDictModule = field(init=False)
    loss_module: DQNLoss = field(init=False)
    target_net_updater: SoftUpdate | None = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def pre_init_hook(self) -> None:
        """Hook for subclasses to optionally run code before __post_init__."""
        pass

    def post_init_hook(self) -> None:
        """Hook for subclasses to optionally run code after __post_init__."""
        pass

    @abstractmethod
    def get_action_value_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    def __post_init__(self) -> None:
        self.pre_init_hook()

        self.action_value_module = self.get_action_value_module().to(self._device)

        self.greedy_module = QValueModule(
            spec=self.action_spec,
            action_mask_key=self.action_mask_key,
            action_value_key="action_value",
            out_keys=["action", "action_value", "chosen_action_value"],
        ).to(self._device)

        self.greedy_policy_module = TensorDictSequential(
            [self.action_value_module, self.greedy_module]
        )

        self.egreedy_module = EGreedyModule(
            spec=self.action_spec,
            action_key="action",
            action_mask_key=self.action_mask_key,
            annealing_num_steps=self.eps_annealing_num_batches,
            eps_init=self.eps_init,
            eps_end=self.eps_end,
        ).to(self._device)

        self.egreedy_policy_module = TensorDictSequential(
            [self.greedy_policy_module, self.egreedy_module]
        )

        self.loss_module = DQNLoss(
            value_network=self.greedy_policy_module,
            loss_function=self.loss_function,
            delay_value=self.delay_value,
            double_dqn=self.double_dqn,
            action_space=self.action_spec,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        if self.delay_value:
            self.target_net_updater = SoftUpdate(
                self.loss_module, eps=1 - self.update_tau
            )
        else:
            self.target_net_updater = None
        self.loss_keys = ["loss"]
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_size, device=self.replay_buffer_device
            ),
            sampler=PrioritizedSampler(
                max_capacity=self.replay_buffer_size,
                alpha=self.replay_buffer_alpha,
                beta=self.replay_buffer_beta_init,
            ),
            # sampler=SamplerWithoutReplacement(),
            priority_key="td_error",
            batch_size=self.replay_buffer_batch_size,
        )

        self.post_init_hook()

    @property
    @override
    def policy(self) -> TensorDictModule:
        return self.egreedy_policy_module

    def _anneal_replay_buffer_beta(self) -> None:
        """Anneal the beta parameter for prioritized sampling."""
        if self.replay_buffer.sampler.beta < self.replay_buffer_beta_end:
            self.replay_buffer.sampler.beta += (
                self.replay_buffer_beta_end - self.replay_buffer_beta_init
            ) / self.replay_buffer_beta_annealing_num_batches

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        # Add to replay buffer
        self.replay_buffer.extend(td)  # type: ignore

        # Only continue if we have enough samples in the replay buffer, in which case we return 0.0 for all losses
        if len(self.replay_buffer) < self.init_random_frames:
            return {k: 0.0 for k in self.loss_keys}

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}
        td_errors = []
        # td_weights = []

        for _ in range(self.num_optim):
            loss_td, td_error = self._sample_and_train()
            td_errors.append(td_error)
            # td_weights.append(td_weight)

            # Accumulate losses
            for k in self.loss_keys:
                total_loss_td[k] += loss_td[k].item()

        # Update target network
        if self.target_net_updater is not None:
            self.target_net_updater.step()

        # Anneal beta for prioritized sampling
        self._anneal_replay_buffer_beta()

        # Anneal epsilon for epsilon greedy exploration
        self.egreedy_module.step()

        # Compute average loss
        process_td = {k: v / self.num_optim for k, v in total_loss_td.items()}
        process_td["td_error"] = torch.mean(torch.stack(td_errors)).item()
        # process_td["td_weight"] = torch.mean(torch.stack(td_weights)).item()

        return process_td

    def _sample_and_train(self) -> tuple[TensorDictBase, Tensor]:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self.device)

        self.optimizer.zero_grad()
        loss_td: TensorDictBase = self.loss_module(td)
        loss_tensor: Tensor = sum(
            (loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device)
        )
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_tensordict_priority(td)  # type: ignore

        return loss_td, td["td_error"]

    @override
    def get_train_info(self) -> dict[str, Any]:
        """Get training information."""
        return {
            # "replay_buffer_beta": self.replay_buffer.sampler.beta,
            "replay_buffer_size": len(self.replay_buffer),
            "eps": self.egreedy_module.eps.item(),
        }

    @property
    @override
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.action_value_module = self.action_value_module.to(self._device)


@dataclass(kw_only=True, eq=False, order=False)
class PPOAgent(Agent, ABC):
    """Proximal Policy Optimization (PPO) agent."""

    action_spec: TensorSpec = unserializable(default=None)

    # Device. All modules created by subclasses will be moved to this device.
    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # PPO parameters
    gamma: float = unserializable(default=1)
    lmbda: float = unserializable(default=0.95)
    clip_epsilon: float = unserializable(default=0.2)  # weight clipping threshold
    entropy_bonus: bool = unserializable(
        default=True
    )  # whether to encourage exploration
    entropy_coef: float = unserializable(
        default=1e-4
    )  # how much to weight the entropy loss term
    critic_coef: float = unserializable(
        default=1.0
    )  # how much to weight the critic loss term
    loss_critic_type: str = unserializable(
        default="smooth_l1"
    )  # what type of loss to use for the critic

    # Optimizer parameters
    lr: float = unserializable(default=1e-3)
    max_grad_norm: float = unserializable(default=1)

    # Replay buffer parameters
    batch_size: int = unserializable(default=1000)
    sub_batch_size: int = unserializable(
        default=100
    )  # size of batch when sampling from replay buffer
    num_epochs: int = unserializable(default=10)
    replay_buffer_device: torch.device = unserializable(
        default_factory=lambda: torch.device("cpu")
    )

    # Set in constructor
    policy_module: ProbabilisticTensorDictSequential = weights(init=False)
    state_value_module: TensorDictModule = weights(init=False)
    advantage_module: GAE = field(init=False)
    loss_module: ClipPPOLoss = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def pre_init_hook(self) -> None:
        """Hook for subclasses to optionally run code before __post_init__."""
        pass

    def post_init_hook(self) -> None:
        """Hook for subclasses to optionally run code after __post_init__."""
        pass

    @abstractmethod
    def get_policy_module(self) -> ProbabilisticTensorDictSequential:
        """Get the policy module."""
        pass

    @abstractmethod
    def get_state_value_module(self) -> TensorDictModule:
        """Get the state value module."""
        pass

    def __post_init__(self) -> None:
        # Ensure batch_size is divisible by sub_batch_size
        if self.batch_size % self.sub_batch_size != 0:
            raise ValueError("batch_size must be divisible by sub_batch_size.")

        self.pre_init_hook()

        self.policy_module = self.get_policy_module().to(self._device)
        self.state_value_module = self.get_state_value_module().to(self._device)

        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.state_value_module,
            average_gae=self.sub_batch_size
            > 1,  # we cannot average or calculate std with a single sample
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.state_value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
        )
        self.loss_keys = ["loss_objective", "loss_critic"] + (
            ["loss_entropy"] if self.entropy_bonus else []
        )
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.batch_size, device=self.replay_buffer_device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.sub_batch_size,
        )

        self.post_init_hook()

    @property
    def policy(self) -> ProbabilisticTensorDictSequential:
        return self.policy_module

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        assert td.batch_size == torch.Size((self.batch_size,)), "Batch size mismatch"

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}

        # Perform multiple epochs of training
        for _ in range(self.num_epochs):
            # Compute advantages each epoch
            self.advantage_module(td)

            # Reset replay buffer each epoch
            self.replay_buffer.extend(td)  # type: ignore

            for _ in range(self.batch_size // self.sub_batch_size):
                loss_td = self._sample_and_train()

                # Accumulate losses
                for k in self.loss_keys:
                    total_loss_td[k] += loss_td[k].item()

        # Compute average loss
        num_updates = self.num_epochs * (self.batch_size // self.sub_batch_size)
        avg_loss_td = {k: v / num_updates for k, v in total_loss_td.items()}

        return avg_loss_td

    def _sample_and_train(self) -> TensorDictBase:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self._device)

        self.optimizer.zero_grad()
        loss_td: TensorDictBase = self.loss_module(td)
        loss_tensor: Tensor = sum(
            (loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device)
        )
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        return loss_td

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.state_value_module = self.state_value_module.to(self._device)
