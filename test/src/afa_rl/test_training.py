"""
RLTrainer runs a short CPU training without touching CUDA.

Builds the real JAFA trainer through its normal constructor from tiny
on-disk bundles, then trains for two batches and calls `finish()`. CUDA is
reported as available so that any device-availability gating (instead of
device-type gating) would reach the recording CUDA stubs.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from afabench.components.initializers.config import InitializerConfig
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
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.utils import get_class_frequencies, set_seed
from afabench.datasets.datasets import CubeDataset
from scripts.train_method.jafa import JAFARLTrainer

SEED = 0
N_BATCHES = 2
EMBEDDING_SIZE = 4


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
        classifier_bundle_path=None,
        save_path="unused.bundle",
        initializer=InitializerConfig(class_name="ZeroInitializer", kwargs={}),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        mdp=AFAMDPConfig(hard_budget=2, force_hard_budget=True, n_agents=2),
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
    trainer = JAFARLTrainer(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        initializer_cfg=cfg.initializer,
        unmasker_cfg=cfg.unmasker,
        mdp_cfg=cfg.mdp,
        n_agents=cfg.mdp.n_agents,
        seed=cfg.seed,
        device=torch.device("cpu"),
        cfg=asdict(cfg),
        use_wandb=cfg.use_wandb,
        typed_cfg=cfg,
    )
    logged: list[dict[str, Any]] = []
    trainer.log_fn = logged.append
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

    trainer.train(cfg=cfg.rl_training_loop)
    trainer.finish()

    assert cuda_calls == []
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
