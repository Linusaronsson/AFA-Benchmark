"""
Testing configurations for Shim2018 RL tests with different unmaskers.

This module provides predefined configurations for testing the Shim2018 RL pipeline
with various unmaskers and environment setups. The configurations are designed to be
lightweight and fast for unit testing while still covering the main functionality.

Available configurations:
- direct_unmasker_1d: Tests with 1D features using DirectUnmasker
- direct_unmasker_2d: Tests with 2D features using DirectUnmasker
- image_patch_unmasker: Tests with image patches using ImagePatchUnmasker

Each configuration includes:
- Feature shapes and dataset parameters
- Unmasker and initializer configurations
- Environment parameters (batch size, hard budget, etc.)
- Pretrain, agent, and train configurations with small networks for fast testing

Usage:
    from afabench.afa_rl.shim2018.shim2018_testing_configs import get_test_config
    config = get_test_config("direct_unmasker_1d")
"""

from dataclasses import dataclass
from typing import Any

import torch

from afabench.common.config_classes import (
    InitializerConfig,
    Shim2018AgentConfig,
    Shim2018ClassifierConfig,
    Shim2018EncoderConfig,
    Shim2018PretrainConfig,
    Shim2018TrainConfig,
    SupervisedLearningConfig,
    UnmaskerConfig,
)


@dataclass
class TestEnvironmentConfig:
    """Configuration for test environment setup."""

    batch_size: int = 2
    hard_budget: int = 3
    n_classes: int = 3
    device: str = "cpu"
    seed: int = 42


@dataclass
class TestDatasetConfig:
    """Configuration for test datasets."""

    n_samples: int = 10
    feature_shape: torch.Size = torch.Size(  # noqa: RUF009
        [4]
    )  # Default 1D features
    noise_std: float = 0.1


def get_direct_unmasker_config() -> UnmaskerConfig:
    """Get configuration for DirectUnmasker (1D features)."""
    return UnmaskerConfig(class_name="DirectUnmasker", kwargs={})


def get_image_patch_unmasker_config() -> UnmaskerConfig:
    """Get configuration for ImagePatchUnmasker (2D image patches)."""
    return UnmaskerConfig(
        class_name="ImagePatchUnmasker",
        kwargs={
            "image_side_length": 8,  # Small 8x8 images for testing
            "patch_size": 2,  # 2x2 patches -> 4x4 grid of patches
            "n_channels": 2,  # 2 channels for testing
        },
    )


def get_cold_initializer_config() -> InitializerConfig:
    """Get configuration for cold start initializer."""
    return InitializerConfig(
        class_name="FixedRandomInitializer", kwargs={"unmask_ratio": 0.0}
    )


def get_warm_initializer_config() -> InitializerConfig:
    """Get configuration for warm start initializer."""
    return InitializerConfig(
        class_name="FixedRandomInitializer", kwargs={"unmask_ratio": 0.2}
    )


def get_basic_pretrain_config() -> Shim2018PretrainConfig:
    """Get basic pretraining configuration for testing."""
    return Shim2018PretrainConfig(
        train_dataset_bundle_path="dummy_train.bundle",
        val_dataset_bundle_path="dummy_val.bundle",
        save_path="dummy_model.bundle",
        device="cpu",
        seed=42,
        supervised_learning=SupervisedLearningConfig(
            batch_size=4,
            max_epochs=1,  # Just 1 epoch for testing
            limit_train_batches=2,
            limit_val_batches=1,
            patience=10,
            val_check_interval=10,
        ),
        lr=1e-3,
        min_masking_probability=0.1,
        max_masking_probability=0.5,
        encoder=Shim2018EncoderConfig(
            output_size=8,
            reading_block_cells=[16, 16],
            writing_block_cells=[16, 16],
            memory_size=8,
            processing_steps=3,
            dropout=0.0,
        ),
        classifier=Shim2018ClassifierConfig(num_cells=[16, 16]),
        use_wandb=False,
    )


def get_basic_agent_config() -> Shim2018AgentConfig:
    """Get basic agent configuration for testing."""
    return Shim2018AgentConfig(
        action_value_num_cells=[16, 16],
        action_value_dropout=0.0,
        eps_init=1.0,
        eps_end=0.1,
        eps_annealing_fraction=0.8,  # 80% of training for annealing
        replay_buffer_batch_size=32,
        lr=1e-3,
        gamma=0.99,
        lmbda=0.95,
        loss_function="l2",
        delay_value=True,
        double_dqn=False,
        update_tau=0.005,
        num_epochs=1,
        max_grad_norm=1.0,
    )


def get_basic_train_config() -> Shim2018TrainConfig:
    """Get basic training configuration for testing."""
    return Shim2018TrainConfig(
        train_dataset_bundle_path="dummy_train.bundle",
        val_dataset_bundle_path="dummy_val.bundle",
        pretrained_model_bundle_path="dummy_pretrained.bundle",
        save_path="dummy_trained.bundle",
        device="cpu",
        seed=42,
        n_agents=2,
        batch_size=4,
        n_batches=5,
        hard_budget=3,
        force_hard_budget=False,
        soft_budget_param=None,
        pretrained_model_lr=1e-4,
        activate_joint_training_after_fraction=0.4,  # 40% through training
        eval_n_times=2,  # Evaluate twice during training
        eval_max_steps=10,
        n_eval_episodes=2,
        agent=get_basic_agent_config(),
        initializer=get_cold_initializer_config(),
        unmasker=get_direct_unmasker_config(),
        use_wandb=False,
    )


# Test configurations for different scenarios
TEST_CONFIGS = {
    "direct_unmasker_1d": {
        "feature_shape": torch.Size([8]),
        "unmasker": get_direct_unmasker_config(),
        "initializer": get_cold_initializer_config(),
        "dataset_config": TestDatasetConfig(
            n_samples=20, feature_shape=torch.Size([8]), noise_std=0.1
        ),
        "env_config": TestEnvironmentConfig(
            batch_size=2, hard_budget=4, n_classes=3
        ),
    },
    "direct_unmasker_2d": {
        "feature_shape": torch.Size([3, 4]),
        "unmasker": get_direct_unmasker_config(),
        "initializer": get_warm_initializer_config(),
        "dataset_config": TestDatasetConfig(
            n_samples=15, feature_shape=torch.Size([3, 4]), noise_std=0.05
        ),
        "env_config": TestEnvironmentConfig(
            batch_size=3, hard_budget=6, n_classes=2
        ),
    },
    "image_patch_unmasker": {
        "feature_shape": torch.Size([2, 8, 8]),
        "unmasker": get_image_patch_unmasker_config(),
        "initializer": get_cold_initializer_config(),
        "dataset_config": TestDatasetConfig(
            n_samples=12, feature_shape=torch.Size([2, 8, 8]), noise_std=0.2
        ),
        "env_config": TestEnvironmentConfig(
            batch_size=2, hard_budget=8, n_classes=4
        ),
    },
}


def get_test_config(config_name: str) -> dict[str, Any]:
    """Get a specific test configuration by name."""
    if config_name not in TEST_CONFIGS:
        available = list(TEST_CONFIGS.keys())
        msg = f"Unknown config '{config_name}'. Available: {available}"
        raise ValueError(msg)
    return TEST_CONFIGS[config_name]


def get_all_test_config_names() -> tuple[str, ...]:
    """Get names of all available test configurations."""
    return tuple(TEST_CONFIGS.keys())
