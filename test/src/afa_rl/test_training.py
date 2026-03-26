from __future__ import annotations

from typing import TYPE_CHECKING, cast, final, override

import torch

from afabench.afa_rl.common.training import RLTrainer
from afabench.common.config_classes import AFARLTrainingLoopConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tensordict import TensorDictBase
    from tensordict.nn import TensorDictModuleBase

    from afabench.afa_rl.common.afa_env import AFAEnv
    from afabench.afa_rl.common.agent_interface import Agent
    from afabench.afa_rl.common.custom_types import AFARewardFn
    from afabench.common.custom_types import AFAMethod


@final
class _FakeAgent:
    _module_device: torch.device

    def __init__(self, device: torch.device) -> None:
        self._module_device = device

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        del td
        return {}

    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, float]:
        del rollout_tds
        return {}

    def get_cheap_info(self) -> dict[str, float]:
        return {}

    def get_expensive_info(self) -> dict[str, float]:
        return {}

    def get_policy(self) -> TensorDictModuleBase:
        return cast("TensorDictModuleBase", object())

    def get_exploitative_policy(self) -> TensorDictModuleBase:
        return cast("TensorDictModuleBase", object())

    def get_exploratory_policy(self) -> TensorDictModuleBase:
        return cast("TensorDictModuleBase", object())

    def get_module_device(self) -> torch.device:
        return self._module_device

    def set_module_device(self, device: torch.device) -> None:
        self._module_device = device

    def get_replay_buffer_device(self) -> torch.device | None:
        return None

    def set_replay_buffer_device(self, device: torch.device) -> None:
        del device


@final
class _StubTrainer(RLTrainer):
    agent: Agent
    train_env: AFAEnv
    device: torch.device

    @override
    def _get_tags(self) -> list[str]:
        raise NotImplementedError

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        raise NotImplementedError

    @override
    def _get_agent(self) -> Agent:
        return self.agent

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        del device
        raise NotImplementedError

    @override
    def _single_collector_step(
        self,
        collector,  # noqa: ANN001
        td: TensorDictBase,
        batch_idx: int,
        cfg: AFARLTrainingLoopConfig,
    ) -> None:
        del collector, td, batch_idx, cfg


def _make_trainer(device: str) -> _StubTrainer:
    trainer = object.__new__(_StubTrainer)
    trainer.train_env = cast("AFAEnv", object())
    trainer.agent = _FakeAgent(torch.device(device))
    trainer.device = torch.device(device)
    return trainer


def _make_training_loop_config() -> AFARLTrainingLoopConfig:
    return AFARLTrainingLoopConfig(
        frames_per_batch=8,
        n_batches=2,
        eval_max_steps=1,
        n_eval_episodes=1,
    )


def test_train_disables_collector_cuda_sync_for_cpu_device(
    monkeypatch,  # noqa: ANN001
) -> None:
    captured_kwargs: dict[str, object] = {}

    class RecordingCollector:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args
            captured_kwargs.update(kwargs)

        def __iter__(self) -> Iterator[None]:
            return iter(())

    monkeypatch.setattr(
        "afabench.afa_rl.common.training.SyncDataCollector",
        RecordingCollector,
    )

    _make_trainer("cpu").train(_make_training_loop_config())

    assert captured_kwargs["device"] == torch.device("cpu")
    assert captured_kwargs["no_cuda_sync"] is True


def test_train_keeps_collector_cuda_sync_for_cuda_device(
    monkeypatch,  # noqa: ANN001
) -> None:
    captured_kwargs: dict[str, object] = {}

    class RecordingCollector:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args
            captured_kwargs.update(kwargs)

        def __iter__(self) -> Iterator[None]:
            return iter(())

    monkeypatch.setattr(
        "afabench.afa_rl.common.training.SyncDataCollector",
        RecordingCollector,
    )

    _make_trainer("cuda").train(_make_training_loop_config())

    assert captured_kwargs["device"] == torch.device("cuda")
    assert captured_kwargs["no_cuda_sync"] is False
