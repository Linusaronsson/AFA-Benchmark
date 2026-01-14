# """
# Comprehensive tests for Shim2018 RL training pipeline with different unmaskers.

# This test suite validates the complete Shim2018 RL pipeline including:

# 1. **Pretrained Model Tests** (`TestShim2018PretrainedModel`):
#    - Creates and tests Shim2018 models without actual training
#    - Tests model forward passes with different feature shapes
#    - Validates Shim2018AFAPredictFn wrapper functionality
#    - Supports 1D features, 2D features, and image patches

# 2. **Agent Tests** (`TestShim2018Agent`):
#    - Creates Shim2018 RL agents with proper configurations
#    - Tests agent policy access (exploratory/exploitative)
#    - Validates agent info methods (cheap/expensive info)

# 3. **Environment Tests** (`TestShim2018Environment`):
#    - Creates AFAEnv with Shim2018 reward functions
#    - Tests environment specs and tensor dict structures
#    - Validates environment step functionality
#    - Tests with DirectUnmasker (1D/2D features) and ImagePatchUnmasker

# 4. **Integration Tests** (`TestShim2018Integration`):
#    - Tests full agent-environment interaction
#    - Validates that agents can successfully take actions and receive rewards
#    - Tests multiple unmasker types in a unified framework

# **Supported Unmasker Types**:
# - DirectUnmasker: For direct feature selection (1D and 2D features)
# - ImagePatchUnmasker: For image patch-based feature acquisition

# **Test Configurations** are defined in `shim2018_testing_configs.py` and include:
# - Feature shapes: [8], [3,4], [2,8,8]
# - Different initializers: cold start (0% features) and warm start (20% features)
# - Various environment parameters: batch sizes, hard budgets, class counts

# These tests ensure that the Shim2018 method can successfully:
# - Create pretrained models for feature embeddings
# - Initialize RL agents with proper Q-networks
# - Interact with MDP environments using different unmasking strategies
# - Process rewards based on classification performance and acquisition costs
# """

# from typing import Any

# import pytest
# import torch
# from jaxtyping import Bool
# from torch import Tensor
# from torchrl.envs import check_env_specs

# from afabench.afa_rl.afa_env import AFAEnv
# from afabench.afa_rl.datasets import get_afa_dataset_fn
# from afabench.afa_rl.shim2018.agents import Shim2018Agent
# from afabench.afa_rl.shim2018.models import (
#     Shim2018AFAPredictFn,
# )
# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
# from afabench.afa_rl.shim2018.shim2018_testing_configs import (
#     get_all_test_config_names,
#     get_basic_pretrain_config,
#     get_test_config,
# )
# from afabench.afa_rl.shim2018.utils import get_shim2018_model_from_config
# from afabench.common.custom_types import (
#     AFAAction,
#     FeatureMask,
#     Features,
#     Label,
#     MaskedFeatures,
#     SelectionMask,
# )
# from afabench.common.initializers.utils import get_afa_initializer_from_config
# from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
# from afabench.common.utils import get_class_frequencies, set_seed


# def create_synthetic_dataset(
#     feature_shape: torch.Size,
#     n_samples: int,
#     n_classes: int,
#     noise_std: float = 0.1,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """Create a synthetic dataset for testing."""
#     # Create deterministic but varied features
#     torch.manual_seed(42)

#     # Generate base patterns for each class
#     features = torch.randn(n_samples, *feature_shape)

#     # Create class-dependent patterns
#     labels_idx = torch.randint(0, n_classes, (n_samples,))

#     # Add class-specific signal to features
#     for class_idx in range(n_classes):
#         class_mask = labels_idx == class_idx
#         if class_mask.any():
#             # Add a class-specific pattern
#             pattern = torch.ones_like(features[0]) * (class_idx + 1) * 0.5
#             features[class_mask] += pattern.unsqueeze(0)

#     # Add noise
#     features += torch.randn_like(features) * noise_std

#     # Convert to one-hot labels
#     labels = torch.zeros(n_samples, n_classes)
#     labels[torch.arange(n_samples), labels_idx] = 1.0

#     return features, labels


# @pytest.fixture
# def device() -> torch.device:
#     """Fixture providing CPU device for tests."""
#     return torch.device("cpu")


# @pytest.fixture(params=get_all_test_config_names())
# def test_config(request: pytest.FixtureRequest) -> dict[str, Any]:
#     """Fixture providing all test configurations."""
#     return get_test_config(request.param)


# class TestShim2018PretrainedModel:
#     """Test creation and basic functionality of pretrained Shim2018 models."""

#     def test_create_pretrained_model(
#         self, device: torch.device, test_config: dict[str, Any]
#     ) -> None:
#         """Test creating a pretrained model without actual training."""
#         set_seed(42)

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]

#         # Create synthetic dataset
#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#             noise_std=dataset_config.noise_std,
#         )

#         # Get class frequencies for model creation
#         class_probabilities = get_class_frequencies(labels)

#         # Create model using config
#         pretrain_config = get_basic_pretrain_config()
#         pretrain_config.encoder.output_size = 12  # Small for testing

#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)
#         lit_model.eval()

#         # Test forward pass
#         batch_size = 3
#         test_features = features[:batch_size].to(device)

#         # Create a simple mask (unmask half the features)
#         feature_mask = torch.rand(*test_features.shape) > 0.5
#         masked_features = test_features.clone()
#         masked_features[~feature_mask] = 0.0

#         # Test model forward pass
#         with torch.no_grad():
#             embedding, logits = lit_model(masked_features, feature_mask)

#         # Verify output shapes
#         assert embedding.shape == (
#             batch_size,
#             pretrain_config.encoder.output_size,
#         )
#         assert logits.shape == (batch_size, env_config.n_classes)

#         # Test that embeddings are different for different inputs
#         feature_mask2 = torch.rand(*test_features.shape) > 0.3
#         masked_features2 = test_features.clone()
#         masked_features2[~feature_mask2] = 0.0

#         with torch.no_grad():
#             embedding2, logits2 = lit_model(masked_features2, feature_mask2)

#         # Embeddings should be different for different masks
#         assert not torch.allclose(embedding, embedding2, atol=1e-6)

#     def test_shim2018_predict_fn(
#         self, device: torch.device, test_config: dict[str, Any]
#     ) -> None:
#         """Test Shim2018AFAPredictFn wrapper."""
#         set_seed(42)

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]

#         # Create model
#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#         )
#         class_probabilities = get_class_frequencies(labels)

#         pretrain_config = get_basic_pretrain_config()
#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)
#         lit_model.eval()

#         # Create predict function wrapper
#         predict_fn = Shim2018AFAPredictFn(lit_model)

#         # Test prediction
#         batch_size = 2
#         test_features = features[:batch_size].to(device)
#         feature_mask = torch.rand(*test_features.shape) > 0.5
#         masked_features = test_features.clone()
#         masked_features[~feature_mask] = 0.0

#         with torch.no_grad():
#             predictions = predict_fn(masked_features, feature_mask)

#         # Verify output
#         assert predictions.shape == (batch_size, env_config.n_classes)
#         assert torch.allclose(
#             predictions.sum(dim=-1), torch.ones(batch_size), atol=1e-6
#         )
#         assert (predictions >= 0).all()
#         assert (predictions <= 1).all()


# class TestShim2018Agent:
#     """Test Shim2018 agent creation and basic functionality."""

#     def test_create_agent(
#         self, device: torch.device, test_config: dict[str, Any]
#     ) -> None:
#         """Test creating a Shim2018 agent."""
#         set_seed(42)

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]
#         unmasker_config = test_config["unmasker"]

#         # Create unmasker to determine n_selections
#         unmasker = get_afa_unmasker_from_config(unmasker_config)
#         n_selections = unmasker.get_n_selections(feature_shape)

#         # Create pretrained model
#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#         )
#         class_probabilities = get_class_frequencies(labels)

#         pretrain_config = get_basic_pretrain_config()
#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)

#         # Create action spec
#         from torchrl.data import Categorical

#         action_spec = Categorical(
#             n=n_selections + 1,
#             shape=torch.Size([env_config.batch_size]),
#             dtype=torch.int64,
#         )

#         # Create agent
#         from afabench.afa_rl.shim2018.shim2018_testing_configs import (
#             get_basic_agent_config,
#         )

#         agent_config = get_basic_agent_config()

#         agent = Shim2018Agent(
#             cfg=agent_config,
#             embedder=lit_model.embedder,
#             embedding_size=pretrain_config.encoder.output_size,
#             action_spec=action_spec,
#             action_mask_key="allowed_action_mask",
#             batch_size=4,  # For processing batch
#             module_device=device,
#             n_feature_dims=len(feature_shape),
#             n_batches=10,  # Total number of batches for testing
#         )

#         # Test that agent was created successfully
#         assert agent.get_module_device() == device

#         # Test policy access
#         exploratory_policy = agent.get_exploratory_policy()
#         exploitative_policy = agent.get_exploitative_policy()

#         assert exploratory_policy is not None
#         assert exploitative_policy is not None

#         # Test info methods
#         cheap_info = agent.get_cheap_info()
#         expensive_info = agent.get_expensive_info()

#         assert isinstance(cheap_info, dict)
#         assert isinstance(expensive_info, dict)
#         assert "eps" in cheap_info


# class TestShim2018Environment:
#     """Test Shim2018 environment integration."""

#     def test_create_environment(
#         self, device: torch.device, test_config: dict[str, Any]
#     ) -> None:
#         """Test creating an environment with Shim2018 components."""
#         set_seed(42)

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]
#         unmasker_config = test_config["unmasker"]
#         initializer_config = test_config["initializer"]

#         # Create components
#         unmasker = get_afa_unmasker_from_config(unmasker_config)
#         initializer = get_afa_initializer_from_config(initializer_config)
#         n_selections = unmasker.get_n_selections(feature_shape)

#         # Create dataset
#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#         )
#         dataset_fn = get_afa_dataset_fn(features, labels, device=device)

#         # Create pretrained model for reward function
#         class_probabilities = get_class_frequencies(labels)
#         pretrain_config = get_basic_pretrain_config()
#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)
#         lit_model.eval()

#         # Create reward function
#         class_weights = 1 / class_probabilities
#         class_weights = (class_weights / class_weights.sum()).to(device)
#         acquisition_costs = 0.1 * torch.ones(n_selections, device=device)

#         reward_fn = get_shim2018_reward_fn(
#             lit_model,
#             class_weights,
#             acquisition_costs,
#         )

#         # Create environment
#         env = AFAEnv(
#             dataset_fn=dataset_fn,
#             reward_fn=reward_fn,
#             device=device,
#             batch_size=torch.Size([env_config.batch_size]),
#             feature_shape=feature_shape,
#             n_selections=n_selections,
#             n_classes=env_config.n_classes,
#             hard_budget=env_config.hard_budget,
#             initialize_fn=initializer.initialize,
#             unmask_fn=unmasker.unmask,
#             seed=env_config.seed,
#         )

#         # Test environment specs
#         check_env_specs(env)

#         # Test environment reset
#         td = env.reset()

#         # Verify tensor dict structure
#         assert "feature_mask" in td
#         assert "performed_action_mask" in td
#         assert "allowed_action_mask" in td
#         assert "performed_selection_mask" in td
#         assert "masked_features" in td
#         assert "features" in td
#         assert "label" in td

#         # Verify shapes
#         batch_size = env_config.batch_size
#         assert (
#             td["feature_mask"].shape
#             == torch.Size([batch_size]) + feature_shape
#         )
#         assert td["performed_action_mask"].shape == torch.Size(
#             [batch_size, n_selections + 1]
#         )
#         assert td["allowed_action_mask"].shape == torch.Size(
#             [batch_size, n_selections + 1]
#         )
#         assert td["performed_selection_mask"].shape == torch.Size(
#             [batch_size, n_selections]
#         )
#         assert (
#             td["masked_features"].shape
#             == torch.Size([batch_size]) + feature_shape
#         )
#         assert td["features"].shape == torch.Size([batch_size]) + feature_shape
#         assert td["label"].shape == torch.Size(
#             [batch_size, env_config.n_classes]
#         )

#     def test_environment_step(
#         self, device: torch.device, test_config: dict[str, Any]
#     ) -> None:
#         """Test environment step functionality."""
#         set_seed(42)

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]
#         unmasker_config = test_config["unmasker"]
#         initializer_config = test_config["initializer"]

#         # Create components (same as above)
#         unmasker = get_afa_unmasker_from_config(unmasker_config)
#         initializer = get_afa_initializer_from_config(initializer_config)
#         n_selections = unmasker.get_n_selections(feature_shape)

#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#         )
#         dataset_fn = get_afa_dataset_fn(features, labels, device=device)

#         class_probabilities = get_class_frequencies(labels)
#         pretrain_config = get_basic_pretrain_config()
#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)
#         lit_model.eval()

#         class_weights = 1 / class_probabilities
#         class_weights = (class_weights / class_weights.sum()).to(device)
#         acquisition_costs = 0.1 * torch.ones(n_selections, device=device)

#         reward_fn = get_shim2018_reward_fn(
#             lit_model,
#             class_weights,
#             acquisition_costs,
#         )

#         env = AFAEnv(
#             dataset_fn=dataset_fn,
#             reward_fn=reward_fn,
#             device=device,
#             batch_size=torch.Size([env_config.batch_size]),
#             feature_shape=feature_shape,
#             n_selections=n_selections,
#             n_classes=env_config.n_classes,
#             hard_budget=env_config.hard_budget,
#             initialize_fn=initializer.initialize,
#             unmask_fn=unmasker.unmask,
#             seed=env_config.seed,
#         )

#         # Test environment step
#         td = env.reset()

#         # Take a valid action (action 1 = select first feature)
#         td["action"] = torch.ones(env_config.batch_size, dtype=torch.int64)

#         td_next = env.step(td)
#         td = td_next["next"]

#         # Verify step results
#         assert "done" in td
#         assert "reward" in td
#         assert td["done"].shape == torch.Size([env_config.batch_size, 1])
#         assert td["reward"].shape == torch.Size([env_config.batch_size, 1])

#         # Verify that performed action mask was updated
#         assert td["performed_action_mask"][
#             :, 1
#         ].all()  # Action 1 should be marked as performed

#         # Test stop action (action 0)
#         td["action"] = torch.zeros(env_config.batch_size, dtype=torch.int64)
#         td_next = env.step(td)
#         td = td_next["next"]

#         # Should be done after stop action
#         assert td["done"].all()


# class TestShim2018Integration:
#     """Test full integration of agent with environment."""

#     def test_agent_environment_interaction(self, device: torch.device) -> None:
#         """Test that agent can interact with environment."""
#         set_seed(42)

#         # Use direct unmasker configuration for simplicity
#         test_config = get_test_config("direct_unmasker_1d")

#         feature_shape = test_config["feature_shape"]
#         env_config = test_config["env_config"]
#         dataset_config = test_config["dataset_config"]
#         unmasker_config = test_config["unmasker"]
#         initializer_config = test_config["initializer"]

#         # Create environment
#         unmasker = get_afa_unmasker_from_config(unmasker_config)
#         initializer = get_afa_initializer_from_config(initializer_config)
#         n_selections = unmasker.get_n_selections(feature_shape)

#         features, labels = create_synthetic_dataset(
#             feature_shape=feature_shape,
#             n_samples=dataset_config.n_samples,
#             n_classes=env_config.n_classes,
#         )
#         dataset_fn = get_afa_dataset_fn(features, labels, device=device)

#         class_probabilities = get_class_frequencies(labels)
#         pretrain_config = get_basic_pretrain_config()
#         lit_model = get_shim2018_model_from_config(
#             pretrain_config,
#             feature_shape=feature_shape,
#             n_classes=env_config.n_classes,
#             class_probabilities=class_probabilities,
#         )
#         lit_model = lit_model.to(device)
#         lit_model.eval()

#         class_weights = 1 / class_probabilities
#         class_weights = (class_weights / class_weights.sum()).to(device)
#         acquisition_costs = 0.1 * torch.ones(n_selections, device=device)

#         reward_fn = get_shim2018_reward_fn(
#             lit_model,
#             class_weights,
#             acquisition_costs,
#         )

#         env = AFAEnv(
#             dataset_fn=dataset_fn,
#             reward_fn=reward_fn,
#             device=device,
#             batch_size=torch.Size([env_config.batch_size]),
#             feature_shape=feature_shape,
#             n_selections=n_selections,
#             n_classes=env_config.n_classes,
#             hard_budget=env_config.hard_budget,
#             initialize_fn=initializer.initialize,
#             unmask_fn=unmasker.unmask,
#             seed=env_config.seed,
#         )

#         # Create agent
#         from torchrl.data import Categorical

#         action_spec = Categorical(
#             n=n_selections + 1,
#             shape=torch.Size([env_config.batch_size]),
#             dtype=torch.int64,
#         )

#         from afabench.afa_rl.shim2018.shim2018_testing_configs import (
#             get_basic_agent_config,
#         )

#         agent_config = get_basic_agent_config()

#         agent = Shim2018Agent(
#             cfg=agent_config,
#             embedder=lit_model.embedder,
#             embedding_size=pretrain_config.encoder.output_size,
#             action_spec=action_spec,
#             action_mask_key="allowed_action_mask",
#             batch_size=4,
#             module_device=device,
#             n_feature_dims=len(feature_shape),
#             n_batches=10,  # Total number of batches for testing
#         )

#         # Test rollout
#         td = env.reset()
#         policy = agent.get_exploitative_policy()

#         # Perform a few steps
#         for _step in range(min(3, env_config.hard_budget)):
#             with torch.no_grad():
#                 td = policy(td)

#             # Verify action is valid
#             assert td["action"].shape == torch.Size([env_config.batch_size])
#             assert (td["action"] >= 0).all()
#             assert (td["action"] <= n_selections).all()

#             # Take step
#             td_next = env.step(td)
#             td = td_next["next"]

#             # If done, break
#             if td["done"].all():
#                 break

#         # Test that agent can successfully interact with environment
#         # This confirms the full integration works
#         assert "done" in td
#         assert "reward" in td
#         assert td["performed_action_mask"].any()  # Some actions were performed

#         # Test agent's cheap and expensive info methods
#         cheap_info = agent.get_cheap_info()
#         expensive_info = agent.get_expensive_info()
#         assert isinstance(cheap_info, dict)
#         assert isinstance(expensive_info, dict)
#         assert "eps" in cheap_info

#     def test_different_unmasker_types(self, device: torch.device) -> None:
#         """Test that different unmasker types work correctly."""
#         set_seed(42)

#         for config_name in get_all_test_config_names():
#             test_config = get_test_config(config_name)

#             feature_shape = test_config["feature_shape"]
#             env_config = test_config["env_config"]
#             dataset_config = test_config["dataset_config"]
#             unmasker_config = test_config["unmasker"]
#             initializer_config = test_config["initializer"]

#             # Create components
#             unmasker = get_afa_unmasker_from_config(unmasker_config)
#             initializer = get_afa_initializer_from_config(initializer_config)
#             n_selections = unmasker.get_n_selections(feature_shape)

#             # Test that n_selections makes sense
#             if unmasker_config.class_name == "DirectUnmasker":
#                 assert n_selections == feature_shape.numel()
#             elif unmasker_config.class_name == "ImagePatchUnmasker":
#                 # For 8x8 image with 2x2 patches, should have 4x4 = 16 patches
#                 expected_patches = (
#                     unmasker_config.kwargs["image_side_length"]
#                     // unmasker_config.kwargs["patch_size"]
#                 ) ** 2
#                 assert n_selections == expected_patches

#             # Create simple dataset and environment to test basic functionality
#             features, labels = create_synthetic_dataset(
#                 feature_shape=feature_shape,
#                 n_samples=dataset_config.n_samples,
#                 n_classes=env_config.n_classes,
#             )
#             dataset_fn = get_afa_dataset_fn(features, labels, device=device)

#             # Create minimal reward function for testing
#             def simple_reward_fn(
#                 _masked_features: MaskedFeatures,  # current masked features
#                 _feature_mask: FeatureMask,  # current feature mask
#                 _selection_mask: SelectionMask,  # current selection mask
#                 _new_masked_features: MaskedFeatures,  # new masked features
#                 _new_feature_mask: FeatureMask,  # new feature mask
#                 _new_selection_mask: SelectionMask,  # new selection mask
#                 _selection: AFAAction,  # which selection we made
#                 _features: Features,  # true features
#                 _label: Label,  # true label
#                 done: Bool[Tensor, "*batch 1"],  # done key
#             ) -> torch.Tensor:
#                 return torch.zeros_like(done, dtype=torch.float32)

#             # Create environment
#             env = AFAEnv(
#                 dataset_fn=dataset_fn,
#                 reward_fn=simple_reward_fn,
#                 device=device,
#                 batch_size=torch.Size([env_config.batch_size]),
#                 feature_shape=feature_shape,
#                 n_selections=n_selections,
#                 n_classes=env_config.n_classes,
#                 hard_budget=env_config.hard_budget,
#                 initialize_fn=initializer.initialize,
#                 unmask_fn=unmasker.unmask,
#                 seed=env_config.seed,
#             )

#             # Test that environment works
#             td = env.reset()

#             # Test valid action
#             td["action"] = torch.ones(env_config.batch_size, dtype=torch.int64)
#             td_next = env.step(td)
#             td = td_next["next"]

#             # Should not raise errors and should produce valid outputs
#             assert "done" in td
#             assert "reward" in td
#             assert (
#                 td["feature_mask"].shape
#                 == torch.Size([env_config.batch_size]) + feature_shape
#             )
