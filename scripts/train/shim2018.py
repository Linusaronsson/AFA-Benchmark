import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from tensordict import TensorDictBase
from torch import optim
from torch.nn import functional as F

from afabench.afa_rl.common.afa_methods import RLAFAMethod
from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.common.custom_types import AFARewardFn
from afabench.afa_rl.common.training import RLTrainer
from afabench.afa_rl.shim2018.agents import Shim2018Agent
from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018AFAClassifier,
)
from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.common.bundle import load_bundle
from afabench.common.config_classes import Shim2018TrainConfig
from afabench.common.custom_types import AFAMethod
from afabench.common.utils import set_seed

if TYPE_CHECKING:
    from afabench.common.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def method_specific_init(
    cfg: Shim2018TrainConfig,
) -> Shim2018TrainConfig:
    """Initialize config specific to Shim2018 training."""
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


class Shim2018RLTrainer(RLTrainer):
    pretrained_model: LitShim2018EmbedderClassifier
    pretrained_model_optim: torch.optim.Adam
    replay_buffer_device: torch.device
    afa_method: RLAFAMethod
    activate_joint_training_after_batch: int
    typed_cfg: Shim2018TrainConfig

    def __init__(
        self,
        *args,  # noqa: ANN002
        typed_cfg: Shim2018TrainConfig,
        **kwargs,  # noqa: ANN003
    ) -> None:
        self.typed_cfg = typed_cfg
        super().__init__(*args, **kwargs)

        self.activate_joint_training_after_batch = int(
            self.typed_cfg.rl_training_loop.n_batches
            * self.typed_cfg.activate_joint_training_after_fraction
        )

    @override
    def _setup_subclass_specific_state(self) -> None:
        self.replay_buffer_device = self.device
        self.pretrained_model, self.pretrained_model_optim = (
            self._get_pretrained_model_and_optim(
                pretrained_model_bundle_path=Path(
                    self.typed_cfg.pretrained_model_bundle_path
                ),
                pretrained_model_lr=self.typed_cfg.pretrained_model_lr,
                device=self.device,
            )
        )

    def _get_pretrained_model_and_optim(
        self,
        pretrained_model_bundle_path: Path,
        pretrained_model_lr: float,
        device: torch.device | None,
    ) -> tuple[LitShim2018EmbedderClassifier, optim.Adam]:
        pretrained_model, _ = load_bundle(
            Path(pretrained_model_bundle_path),
            device=device,
        )
        torch_model_bundle = cast(
            "TorchModelBundle",
            cast("object", pretrained_model),
        )
        pretrained_model = cast(
            "LitShim2018EmbedderClassifier", torch_model_bundle.model
        )
        pretrained_model.eval()
        pretrained_model = pretrained_model.to(device)
        pretrained_model_optim = optim.Adam(
            pretrained_model.parameters(), lr=pretrained_model_lr
        )
        return pretrained_model, pretrained_model_optim

    @override
    def _get_tags(self) -> list[str]:
        return ["shim2018"]

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        return get_shim2018_reward_fn(
            pretrained_model=self.pretrained_model,
            weights=self.class_weights,
            selection_costs=(
                0
                if self.typed_cfg.soft_budget_param is None
                else self.typed_cfg.soft_budget_param
            )
            * self.normalized_selection_costs.to(self.device),
            n_feature_dims=self._n_feature_dims,
        )

    @override
    def _get_agent(self) -> Agent:
        is_hard_budget_mode = self.typed_cfg.soft_budget_param is None
        return Shim2018Agent(
            cfg=self.typed_cfg.agent,
            embedder=self.pretrained_model.embedder,
            embedding_size=self.pretrained_model.embedder.encoder.output_size,
            action_spec=self.train_env.action_spec,
            action_mask_key="allowed_action_mask",
            module_device=self.device,
            n_feature_dims=len(self.train_dataset.feature_shape),
            n_batches=self.typed_cfg.rl_training_loop.n_batches,
            allow_stop_action=is_hard_budget_mode,
        )

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        return RLAFAMethod(
            self.agent.get_exploitative_policy().to(device),
            Shim2018AFAClassifier(self.pretrained_model, device=device),
            device,
        )

    @override
    def _post_process_batch(
        self, td: TensorDictBase, batch_idx: int
    ) -> dict[str, Any]:
        assert td.batch_dims == 2, "Expected two batch dimensions"

        if batch_idx >= self.activate_joint_training_after_batch:
            if batch_idx == self.activate_joint_training_after_batch:
                log.info(
                    "Activating joint training of classifier and embedder"
                )
            self.pretrained_model.train()
            self.pretrained_model_optim.zero_grad()

            n_feature_dims = len(self.train_dataset.feature_shape)

            # Flatten feature dims
            flat_masked_features = td["next", "masked_features"].flatten(
                start_dim=-n_feature_dims
            )
            flat_feature_mask = td["next", "feature_mask"].flatten(
                start_dim=-n_feature_dims
            )
            assert flat_masked_features.ndim == td["next", "label"].ndim, (
                "Label should be 1D"
            )

            # Flatten batch dims
            flat_masked_features = flat_masked_features.flatten(end_dim=-2)
            flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
            flat_label = td["next", "label"].flatten(end_dim=-2)

            _, logits_next = self.pretrained_model(
                flat_masked_features, flat_feature_mask
            )
            class_loss_next = F.cross_entropy(
                logits_next,
                flat_label,
                weight=self.class_weights,
            )
            class_loss_next.mean().backward()

            self.pretrained_model_optim.step()
            self.pretrained_model.eval()

            return {"avg_class_loss": class_loss_next.mean().cpu().item()}
        return {}

    @override
    def _pre_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.eval_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def _post_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.train_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/shim2018",
    config_name="config",
)
def main(cfg: Shim2018TrainConfig) -> None:
    cfg = method_specific_init(cfg)

    trainer = Shim2018RLTrainer(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        initializer_cfg=cfg.initializer,
        unmasker_cfg=cfg.unmasker,
        mdp_cfg=cfg.mdp,
        n_agents=cfg.mdp.n_agents,
        seed=cfg.seed,
        device=cfg.device if cfg.device is not None else torch.device("cpu"),
        cfg=cast("dict[str,Any]", OmegaConf.to_container(cfg)),
        use_wandb=cfg.use_wandb,
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
