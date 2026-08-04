import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from tensordict import TensorDictBase
from torch import optim
from torch.nn import functional as F

from afabench.components.methods.rl.common.afa_methods import RLAFAMethod
from afabench.components.methods.rl.common.agent_interface import Agent
from afabench.components.methods.rl.common.custom_types import (
    AFAFeatureRestorationFn,
    AFARewardFn,
)
from afabench.components.methods.rl.common.training import (
    RLTrainer,
)
from afabench.components.methods.rl.odin.models import ODINPretrainingModel
from afabench.components.methods.rl.ol.agents import (
    OLAgent,
)
from afabench.components.methods.rl.ol.config import OLTrainConfig
from afabench.components.methods.rl.ol.models import (
    LitOLPQModule,
    OLAFAClassifier,
)
from afabench.components.methods.rl.ol.reward import (
    get_ol_reward_fn,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import AFAMethod
from afabench.core.utils import (
    set_seed,
)
from afabench.missing_values.restoration import (
    PVAEStepwiseRestorer,
    load_pvae,
)

if TYPE_CHECKING:
    from afabench.core.bundle_system.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def method_specific_init(
    cfg: OLTrainConfig,
) -> OLTrainConfig:
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


class OLRLTrainer(RLTrainer):
    pretrained_model: LitOLPQModule
    pretrained_model_optim: torch.optim.Adam
    replay_buffer_device: torch.device
    afa_method: RLAFAMethod
    activate_joint_training_after_batch: int
    typed_cfg: OLTrainConfig
    stepwise_pvae: ODINPretrainingModel | None

    def __init__(
        self,
        *args,  # noqa: ANN002
        typed_cfg: OLTrainConfig,
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
        self.replay_buffer_device = (
            self.device
            if self.typed_cfg.replay_buffer_device_same_as_device
            else torch.device("cpu")
        )
        self.pretrained_model, self.pretrained_model_optim = (
            self._get_pretrained_model_and_optim(
                pretrained_model_bundle_path=Path(
                    self.typed_cfg.pretrained_model_bundle_path
                ),
                pretrained_model_lr=self.typed_cfg.pretrained_model_lr,
                device=self.device,
            )
        )
        self.stepwise_pvae = (
            None
            if self.typed_cfg.stepwise_pvae_bundle_path is None
            else load_pvae(
                Path(self.typed_cfg.stepwise_pvae_bundle_path),
                self.device,
            )
        )

    @override
    def _get_feature_restoration_fn(
        self,
        *,
        validation: bool,
    ) -> AFAFeatureRestorationFn | None:
        if self.stepwise_pvae is None:
            return None
        seed = (self.seed or 0) + int(validation)
        return PVAEStepwiseRestorer(
            self.stepwise_pvae,
            n_classes=self._n_classes,
            seed=seed,
        )

    def _get_pretrained_model_and_optim(
        self,
        pretrained_model_bundle_path: Path,
        pretrained_model_lr: float,
        device: torch.device | None,
    ) -> tuple[LitOLPQModule, optim.Adam]:
        pretrained_model, _ = load_bundle(
            Path(pretrained_model_bundle_path),
            device=device,
        )
        torch_model_bundle = cast(
            "TorchModelBundle",
            cast("object", pretrained_model),
        )
        pretrained_model = cast("LitOLPQModule", torch_model_bundle.model)
        pretrained_model.eval()
        pretrained_model = pretrained_model.to(device)
        pretrained_model_optim = optim.Adam(
            pretrained_model.parameters(), lr=pretrained_model_lr
        )
        return pretrained_model, pretrained_model_optim

    @override
    def _get_tags(self) -> list[str]:
        return ["ol"]

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        return get_ol_reward_fn(
            pretrained_model=self.pretrained_model.pq_module,
            selection_costs=self.normalized_selection_costs.to(self.device),
            n_feature_dims=self._n_feature_dims,
            method=self.typed_cfg.reward_method,
            mcdrop_samples=self.typed_cfg.mcdrop_samples,
        )

    @override
    def _get_agent(self) -> Agent:
        return OLAgent(
            cfg=self.typed_cfg.agent,
            pq_module=self.pretrained_model.pq_module,
            action_spec=self.train_env.action_spec_unbatched,
            action_mask_key="allowed_action_mask",
            module_device=self.device,
            replay_buffer_device=self.replay_buffer_device,
            n_feature_dims=len(self.train_dataset.feature_shape),
            n_batches=self.typed_cfg.rl_training_loop.n_batches,
            collect_metrics=self.use_wandb,
        )

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        return RLAFAMethod(
            self.agent.get_exploitative_policy().to(device),
            OLAFAClassifier(self.pretrained_model.pq_module, device=device),
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
                    "Starting classifier fine-tuning alongside policy "
                    "training at batch %d",
                    batch_idx,
                )
            self.pretrained_model.pq_module.train()
            self.pretrained_model_optim.zero_grad()

            # Flatten feature dims
            masked_features = cast(
                "torch.Tensor", td.get(("next", "masked_features"))
            )
            feature_mask = cast(
                "torch.Tensor", td.get(("next", "feature_mask"))
            )
            label = cast("torch.Tensor", td.get(("next", "label")))
            flat_masked_features = masked_features.flatten(
                start_dim=-self._n_feature_dims
            )
            flat_feature_mask = feature_mask.flatten(
                start_dim=-self._n_feature_dims
            )
            assert flat_masked_features.ndim == label.ndim

            # Flatten batch dims
            flat_masked_features = flat_masked_features.flatten(end_dim=-2)
            flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
            flat_label = label.flatten(end_dim=-2)

            logits_next, _qvalues = self.pretrained_model.pq_module.forward(
                flat_masked_features, flat_feature_mask
            )
            class_loss_next = F.cross_entropy(
                logits_next,
                flat_label,
                weight=self.class_weights,
            )
            class_loss_next.mean().backward()

            self.pretrained_model_optim.step()
            self.pretrained_model.pq_module.eval()

            if self.use_wandb:
                return {"avg_class_loss": class_loss_next.mean().cpu().item()}
            return {}
        return {}

    @override
    def _pre_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.eval_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def _post_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.train_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/ol",
    config_name="config",
)
def main(cfg: OLTrainConfig) -> None:
    cfg = cast("OLTrainConfig", OmegaConf.to_object(cfg))
    cfg = method_specific_init(cfg)

    trainer = OLRLTrainer(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        initializer_cfg=cfg.initializer,
        unmasker_cfg=cfg.unmasker,
        mdp_cfg=cfg.mdp,
        n_agents=cfg.mdp.n_agents,
        seed=cfg.seed,
        device=cfg.device if cfg.device is not None else torch.device("cpu"),
        cfg=asdict(cfg),
        use_wandb=cfg.use_wandb,
        typed_cfg=cfg,
    )

    try:
        trainer.train(cfg=cfg.rl_training_loop)
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        raise

    log.info("Training completed, saving model")
    trainer.save(save_path=Path(cfg.save_path))
    log.info("Script completed successfully")


if __name__ == "__main__":
    main()
