"""
RLTrainer runs a short CPU training without touching CUDA.

Builds the real JAFA trainer through its normal constructor from tiny
on-disk bundles, trains for two batches and returns the trained AFA method.
CUDA is reported as available so that any device-availability gating
(instead of device-type gating) would reach the recording CUDA stubs.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.rl.common.afa_methods import RLAFAMethod
from afabench.components.methods.rl.common.config import (
    AFAMDPConfig,
    AFARLTrainingLoopConfig,
)
from afabench.components.methods.rl.jafa.config import (
    JAFAAgentConfig,
    JAFATrainConfig,
)
from afabench.components.methods.rl.jafa.models import (
    JAFAEmbedder,
    JAFAMLPClassifier,
    LitJAFAEmbedderClassifier,
    ReadProcessEncoder,
)
from afabench.components.methods.rl.jafa.training import (
    JAFARLTrainer,
    train_jafa,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.utils import get_class_frequencies, set_seed
from afabench.datasets.datasets import CubeDataset
from afabench.training.inputs import load_inputs
from afabench.training.smoke_test import SMOKE_TEST_N_BATCHES

SEED = 0
N_BATCHES = 2
EMBEDDING_SIZE = 4


class RecordingMetricLogger:
    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []

    def log(self, metrics: Mapping[str, object]) -> None:
        self.metrics.append(dict(metrics))

    def finish(self) -> None:
        pass


def _save_tiny_bundles(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_dataset = CubeDataset(seed=SEED, n_samples=64)
    val_dataset = CubeDataset(seed=SEED + 1, n_samples=16)
    train_path = tmp_path / "train.bundle"
    val_path = tmp_path / "val.bundle"
    save_bundle(train_dataset, train_path, {})
    save_bundle(val_dataset, val_path, {})

    _features, train_labels = train_dataset.get_all_data()
    n_features = train_dataset.feature_shape.numel()
    pretrained_model = LitJAFAEmbedderClassifier(
        embedder=JAFAEmbedder(
            ReadProcessEncoder(
                set_element_size=n_features + 1,
                output_size=EMBEDDING_SIZE,
                reading_block_cells=(8,),
                writing_block_cells=(8,),
                memory_size=4,
                processing_steps=1,
            )
        ),
        classifier=JAFAMLPClassifier(
            EMBEDDING_SIZE, train_dataset.label_shape.numel(), (8,)
        ),
        class_probabilities=get_class_frequencies(train_labels),
    )
    pretrained_path = tmp_path / "pretrained.bundle"
    save_bundle(TorchModelBundle(pretrained_model), pretrained_path, {})
    return train_path, val_path, pretrained_path


def _make_jafa_train_config(
    train_path: Path, val_path: Path, pretrained_path: Path
) -> JAFATrainConfig:
    return JAFATrainConfig(
        train_dataset_bundle_path=str(train_path),
        val_dataset_bundle_path=str(val_path),
        pretrained_model_bundle_path=str(pretrained_path),
        classifier_bundle_path="unused-classifier.bundle",
        save_path="unused.bundle",
        initializer=InitializerConfig(class_name="ZeroInitializer", kwargs={}),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        hard_budget=2,
        mdp=AFAMDPConfig(force_hard_budget=True, n_agents=2),
        rl_training_loop=AFARLTrainingLoopConfig(
            frames_per_batch=4,
            n_batches=N_BATCHES,
            eval_max_steps=3,
            n_eval_episodes=1,
            eval_n_times=N_BATCHES,
        ),
        soft_budget_param=None,
        agent=JAFAAgentConfig(
            eps_init=1.0,
            eps_end=0.1,
            eps_annealing_fraction=1.0,
            num_epochs=1,
            max_grad_norm=1.0,
            lr=1e-2,
            update_tau=0.005,
            action_value_num_cells=[8],
            action_value_dropout=0.0,
            loss_function="l2",
            delay_value=True,
            double_dqn=True,
            gamma=1.0,
            lmbda=0.75,
        ),
        pretrained_model_lr=1e-3,
        activate_joint_training_after_fraction=0.0,
        seed=SEED,
        use_wandb=False,
        device="cpu",
    )


def test_rl_trainer_trains_on_cpu_without_touching_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_seed(SEED)
    cfg = _make_jafa_train_config(*_save_tiny_bundles(tmp_path))
    metric_logger = RecordingMetricLogger()
    trainer = JAFARLTrainer(cfg, load_inputs(cfg), metric_logger)
    value_net_before = [
        p.detach().clone()
        for p in trainer.agent.action_value_module.net.parameters()  # pyright: ignore[reportAttributeAccessIssue]
    ]

    cuda_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Queried by torch.optim once CUDA reports as available
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *_: cuda_calls.append("sync")
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: cuda_calls.append("empty_cache")
    )

    afa_method = trainer.train(cfg=cfg.rl_training_loop)

    assert cuda_calls == []
    assert isinstance(afa_method, RLAFAMethod)
    logged = metric_logger.metrics
    assert len(logged) == N_BATCHES
    assert all("train/agent_process_batch_info.loss" in d for d in logged)
    assert "train/post_process_info.avg_class_loss" in logged[0]
    assert any(k.startswith("eval/metrics.") for k in logged[-1])
    value_net_after = list(
        trainer.agent.action_value_module.net.parameters()  # pyright: ignore[reportAttributeAccessIssue]
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            value_net_before, value_net_after, strict=True
        )
    )


def test_train_jafa_limits_a_smoke_test_to_a_few_batches(
    tmp_path: Path,
) -> None:
    set_seed(SEED)
    cfg = _make_jafa_train_config(*_save_tiny_bundles(tmp_path))
    cfg = replace(
        cfg,
        smoke_test=True,
        rl_training_loop=replace(
            cfg.rl_training_loop, n_batches=50, eval_n_times=0
        ),
    )
    metric_logger = RecordingMetricLogger()

    afa_method = train_jafa(cfg, load_inputs(cfg), metric_logger)

    assert isinstance(afa_method, RLAFAMethod)
    assert len(metric_logger.metrics) == SMOKE_TEST_N_BATCHES
