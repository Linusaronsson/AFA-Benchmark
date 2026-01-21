import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from torch.nn import functional as F

from afabench.afa_rl.common.afa_methods import RLAFAMethod
from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.common.custom_types import AFARewardFn
from afabench.afa_rl.common.training import RLTrainer
from afabench.afa_rl.zannone2019.agents import Zannone2019Agent
from afabench.afa_rl.zannone2019.models import (
    Zannone2019AFAClassifier,
    Zannone2019PretrainingModel,
)
from afabench.afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afabench.common.bundle import load_bundle
from afabench.common.config_classes import Zannone2019TrainConfig
from afabench.common.custom_types import AFAMethod, Features, Label
from afabench.common.datasets.wrappers import ExtendedAFADataset
from afabench.common.utils import set_seed

if TYPE_CHECKING:
    from afabench.common.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def method_specific_init(
    cfg: Zannone2019TrainConfig,
) -> Zannone2019TrainConfig:
    """Initialize config specific to Zannone2019 training."""
    # Evaluate alias arguments
    # Flat hard budget parameter always overrides
    cfg.mdp.hard_budget = cfg.hard_budget

    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.rl_training_loop.n_batches = 2

    return cfg


def generate_data_batched(
    pretrained_model: Zannone2019PretrainingModel,
    samples: int,
    batch_size: int,
) -> tuple[Features, Label]:
    """Generate synthetic data using the generative model in batches."""
    generated_flat_features = torch.zeros(samples, pretrained_model.n_features)
    generated_labels = torch.zeros(samples, pretrained_model.n_classes)
    n_full_batches = samples // batch_size
    n_samples_rest = samples % batch_size
    # Add full batches
    batch_plan = [
        (i * batch_size, (i + 1) * batch_size, batch_size)
        for i in range(n_full_batches)
    ]
    # Add remainder batch
    if n_samples_rest > 0:
        batch_plan.append(
            (n_full_batches * batch_size, samples, n_samples_rest)
        )

    for start, end, curr_batch_size in batch_plan:
        _z, flat_batch, label_batch = pretrained_model.generate_data(
            n_samples=curr_batch_size
        )
        generated_flat_features[start:end, :] = flat_batch.cpu()
        generated_labels[start:end, :] = F.one_hot(
            label_batch.argmax(-1),
            num_classes=label_batch.shape[-1],
        ).cpu()
    return generated_flat_features, generated_labels


class Zannone2019RLTrainer(RLTrainer):
    pretrained_model: Zannone2019PretrainingModel
    extended_train_dataset: ExtendedAFADataset | Any
    typed_cfg: Zannone2019TrainConfig

    def __init__(
        self,
        *args,  # noqa: ANN002
        typed_cfg: Zannone2019TrainConfig,
        **kwargs,  # noqa: ANN003
    ) -> None:
        self.typed_cfg = typed_cfg
        super().__init__(*args, **kwargs)

    @override
    def _setup_subclass_specific_state(self) -> None:
        """Load pretrained model and generate synthetic data if needed."""
        self.pretrained_model = self._get_pretrained_model(
            pretrained_model_bundle_path=Path(
                self.typed_cfg.pretrained_model_bundle_path
            ),
            device=self.device,
        )

        # zannone2019 unique step: generate additional data using generative model
        if self.typed_cfg.additional_generation_fraction > 0.0:
            n_artificial_samples = int(
                self.typed_cfg.additional_generation_fraction
                * len(self.train_dataset)
            )
            additional_features, additional_labels = generate_data_batched(
                pretrained_model=self.pretrained_model,
                samples=n_artificial_samples,
                batch_size=self.typed_cfg.generation_batch_size,
            )
            self.extended_train_dataset = ExtendedAFADataset(
                base_dataset=self.train_dataset,
                additional_features=additional_features,
                additional_labels=additional_labels,
            )
        else:
            self.extended_train_dataset = self.train_dataset

    def _get_pretrained_model(
        self,
        pretrained_model_bundle_path: Path,
        device: torch.device,
    ) -> Zannone2019PretrainingModel:
        """Load the pretrained generative model."""
        pretrained_model, _ = load_bundle(
            Path(pretrained_model_bundle_path),
            device=device,
        )
        torch_model_bundle = cast(
            "TorchModelBundle",
            cast("object", pretrained_model),
        )
        pretrained_model = cast(
            "Zannone2019PretrainingModel", torch_model_bundle.model
        )
        pretrained_model.eval()
        pretrained_model = pretrained_model.to(device)
        return pretrained_model

    @override
    def _get_tags(self) -> list[str]:
        return ["zannone2019"]

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        return get_zannone2019_reward_fn(
            pretrained_model=self.pretrained_model,
            weights=self.class_weights,
            selection_costs=(
                0
                if self.typed_cfg.soft_budget_param is None
                else self.typed_cfg.soft_budget_param
            )
            * self.unnormalized_selection_costs.to(self.device),
            n_feature_dims=self._n_feature_dims,
        )

    @override
    def _get_agent(self) -> Agent:
        return Zannone2019Agent(
            cfg=self.typed_cfg.agent,
            pointnet=self.pretrained_model.partial_vae.pointnet,
            encoder=self.pretrained_model.partial_vae.encoder,
            action_spec=self.train_env.action_spec,
            latent_size=self.pretrained_model.latent_size,
            action_mask_key="allowed_action_mask",
            frames_per_batch=self.typed_cfg.rl_training_loop.frames_per_batch,
            module_device=self.device,
            n_feature_dims=len(self.extended_train_dataset.feature_shape),
        )

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        return RLAFAMethod(
            self.agent.get_exploitative_policy().to(device),
            Zannone2019AFAClassifier(self.pretrained_model, device=device),
        )

    @override
    def _create_envs(self) -> None:
        """Create environments using the extended training dataset."""
        self.train_env = self._get_env_from_dataset(  # pyright: ignore[reportUnannotatedClassAttribute]
            self.extended_train_dataset
        )
        self.eval_env = self._get_env_from_dataset(self.val_dataset)  # pyright: ignore[reportUnannotatedClassAttribute]


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019TrainConfig) -> None:
    cfg = method_specific_init(cfg)

    trainer = Zannone2019RLTrainer(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        initializer_cfg=cfg.initializer,
        unmasker_cfg=cfg.unmasker,
        mdp_cfg=cfg.mdp,
        n_agents=cfg.mdp.n_agents,
        seed=cfg.seed,
        device=cfg.device if cfg.device is not None else torch.device("cpu"),
        cfg=cast("dict[str,Any]", OmegaConf.to_container(cfg)),
        typed_cfg=cfg,
    )

    try:
        trainer.train(cfg=cfg.rl_training_loop)
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    finally:
        log.info("Training completed, starting cleanup and model saving")
        trainer.save(save_path=Path(cfg.save_path))
        log.info("Script completed successfully")


if __name__ == "__main__":
    main()
