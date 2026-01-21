# TODO: update
# import pytest
# import torch

# from afabench.afa_rl.afa_env import AFAEnv
# from afabench.common.afa_initializers import (
#     FixedRandomInitializer,
#     LeastInformativeInitializer,
#     MutualInformationInitializer,
#     RandomPerEpisodeInitializer,
# )
# from afabench.common.config_classes import (
#     FixedRandomInitializerConfig,
#     LeastInformativeInitializerConfig,
#     MutualInformationInitializerConfig,
#     RandomPerEpisodeInitializerConfig,
# )
# from afabench.common.datasets.datasets import CubeDataset


# @pytest.fixture
# def cube_dataset() -> CubeDataset:
#     """Generate CUBE dataset locally."""
#     dataset = CubeDataset(n_samples=100, seed=42)
#     return dataset


# @pytest.fixture
# def dataset_fn_factory() -> callable:
#     """Factory to create dataset_fn like in training scripts."""

#     def make_dataset_fn(
#         features: torch.Tensor, labels: torch.Tensor
#     ) -> callable:
#         def dataset_fn(
#             batch_size: torch.Size,
#         ) -> tuple[torch.Tensor, torch.Tensor]:
#             idx = torch.randint(
#                 0,
#                 len(features),
#                 (batch_size[0],),
#                 generator=torch.default_generator,
#             )
#             return features[idx], labels[idx]

#         return dataset_fn

#     return make_dataset_fn


# @pytest.fixture
# def simple_reward_fn() -> callable:
#     """Simple reward function."""

#     def reward_fn(td: dict) -> torch.Tensor:
#         n_features = td["feature_mask"].sum(dim=-1, keepdim=True)
#         return -n_features.to(torch.float32)

#     return reward_fn


# class TestWarmStartWithCUBE:
#     """Test warm-start with CUBE dataset."""

#     def test_fixed_strategy_with_cube(self, cube_dataset: CubeDataset) -> None:
#         """Test FixedRandomInitializer with CUBE."""
#         config = FixedRandomInitializerConfig(
#             unmask_ratio=2 / cube_dataset.n_features
#         )
#         initializer = FixedRandomInitializer(config)
#         initializer.set_seed(42)

#         mask = initializer.initialize(
#             features=cube_dataset.features,
#             feature_shape=torch.Size([cube_dataset.n_features]),
#         )

#         assert mask.sum() == 2
#         assert mask.shape == torch.Size([cube_dataset.n_features])

#     def test_mi_strategy_with_cube(self, cube_dataset: CubeDataset) -> None:
#         """Test MutualInformationInitializer with CUBE."""
#         config = MutualInformationInitializerConfig(
#             unmask_ratio=3 / cube_dataset.n_features
#         )
#         initializer = MutualInformationInitializer(config)
#         initializer.set_seed(42)

#         mask = initializer.initialize(
#             features=cube_dataset.features,
#             label=cube_dataset.labels.argmax(dim=-1),
#             feature_shape=torch.Size([cube_dataset.n_features]),
#         )

#         assert mask.sum() == 3
#         # CUBE: features 0-9 informative, 10-19 dummy
#         informative_selected = sum(
#             1 for idx in mask.nonzero(as_tuple=True)[0] if idx < 10
#         )
#         assert informative_selected >= 2

#     def test_least_informative_with_cube(
#         self, cube_dataset: CubeDataset
#     ) -> None:
#         """Test LeastInformativeInitializer with CUBE."""
#         config = LeastInformativeInitializerConfig(
#             unmask_ratio=3 / cube_dataset.n_features
#         )
#         initializer = LeastInformativeInitializer(config)
#         initializer.set_seed(42)

#         mask = initializer.initialize(
#             features=cube_dataset.features,
#             label=cube_dataset.labels.argmax(dim=-1),
#             feature_shape=torch.Size([cube_dataset.n_features]),
#         )

#         assert mask.sum() == 3
#         # Should select mostly dummy features (10-19)
#         dummy_selected = sum(
#             1 for idx in mask.nonzero(as_tuple=True)[0] if idx >= 10
#         )
#         assert dummy_selected >= 2

#     def test_random_per_episode_with_cube(
#         self, cube_dataset: CubeDataset
#     ) -> None:
#         """Test RandomPerEpisodeInitializer with CUBE."""
#         config = RandomPerEpisodeInitializerConfig(
#             unmask_ratio=2 / cube_dataset.n_features
#         )
#         initializer = RandomPerEpisodeInitializer(config)
#         initializer.set_seed(42)

#         masks = [
#             initializer.initialize(
#                 features=cube_dataset.features,
#                 feature_shape=torch.Size([cube_dataset.n_features]),
#             )
#             for _ in range(5)
#         ]

#         for mask in masks:
#             assert mask.sum() == 2
#             assert mask.shape == torch.Size([cube_dataset.n_features])

#         unique_masks = {
#             tuple(mask.nonzero(as_tuple=True)[0].tolist()) for mask in masks
#         }
#         assert len(unique_masks) >= 2


# class TestAFAEnvIntegration:
#     """Test warm-start workflow with AFAEnv."""

#     def test_afaenv_without_warmstart(
#         self, cube_dataset, dataset_fn_factory, simple_reward_fn
#     ) -> None:
#         """Baseline: AFAEnv works without warm-start."""
#         dataset_fn = dataset_fn_factory(
#             cube_dataset.features, cube_dataset.labels
#         )

#         env = AFAEnv(
#             dataset_fn=dataset_fn,
#             reward_fn=simple_reward_fn,
#             device=torch.device("cpu"),
#             batch_size=torch.Size((2,)),
#             feature_size=cube_dataset.n_features,
#             n_classes=cube_dataset.n_classes,
#             hard_budget=5,
#         )

#         td = env.reset()
#         assert td["feature_mask"].sum() == 0

#     def test_warm_start_manual_application(
#         self, cube_dataset, dataset_fn_factory, simple_reward_fn
#     ) -> None:
#         """Test manually applying warm-start to feature masks."""
#         config = FixedRandomInitializerConfig(
#             unmask_ratio=2 / cube_dataset.n_features
#         )
#         initializer = FixedRandomInitializer(config)
#         initializer.set_seed(42)

#         mask = initializer.initialize(
#             features=cube_dataset.features,
#             feature_shape=torch.Size([cube_dataset.n_features]),
#         )
#         selected = mask.nonzero(as_tuple=True)[0].tolist()

#         batch_size = 3
#         feature_mask = torch.zeros(
#             batch_size, cube_dataset.n_features, dtype=torch.bool
#         )
#         for idx in selected:
#             feature_mask[:, idx] = True

#         assert feature_mask.sum() == batch_size * 2
#         for idx in selected:
#             assert feature_mask[:, idx].all()


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
