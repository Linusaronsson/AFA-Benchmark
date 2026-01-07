import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from tensordict import TensorDict
from torch import optim
from torch.nn import functional as F

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.afa_methods import RLAFAMethod
from afabench.afa_rl.common.training import (
    afa_rl_training_loop,
    afa_rl_training_prep,
    create_afa_envs,
)

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
from afabench.afa_rl.shim2018.agents import Shim2018Agent
from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018AFAClassifier,
    Shim2018AFAPredictFn,
)

# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    Shim2018TrainConfig,
)
from afabench.common.utils import (
    initialize_wandb_run,
    set_seed,
)

if TYPE_CHECKING:
    from afabench.common.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def get_post_process_batch_callback(
    pretrained_model: LitShim2018EmbedderClassifier,
    pretrained_model_optim: optim.Adam,
    activate_joint_training_after_batch: int,
    class_weights: torch.Tensor,
    feature_shape: torch.Size,
) -> Callable[[TensorDict, int], dict[str, Any]]:
    def f(td: TensorDict, batch_idx: int) -> dict[str, Any]:
        # Train classifier and embedder jointly if we have reached the correct batch
        assert td.batch_dims == 2, "Expected two batch dimensions"

        if batch_idx >= activate_joint_training_after_batch:
            if batch_idx == activate_joint_training_after_batch:
                log.info(
                    "Activating joint training of classifier and embedder"
                )
            pretrained_model.train()
            pretrained_model_optim.zero_grad()

            n_feature_dims = len(feature_shape)

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

            _, logits_next = pretrained_model(
                flat_masked_features, flat_feature_mask
            )
            class_loss_next = F.cross_entropy(
                logits_next,
                flat_label,
                weight=class_weights,
            )
            class_loss_next.mean().backward()

            pretrained_model_optim.step()
            pretrained_model.eval()

            return {"avg_class_loss": class_loss_next.mean().cpu().item()}
        return {}

    return f


def get_shim2018_pretrained_model(
    pretrained_model_bundle_path: Path,
    pretrained_model_lr: float,
    device: torch.device,
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


def get_pre_eval_callback(
    agent: Shim2018Agent, eval_env: AFAEnv
) -> Callable[[], None]:
    def f() -> None:
        # HACK: Set the action spec of the agent to the eval env action spec
        agent.egreedy_tdmodule._spec = eval_env.action_spec  # noqa: SLF001

    return f


def get_post_eval_callback(
    agent: Shim2018Agent, train_env: AFAEnv
) -> Callable[[], None]:
    def f() -> None:
        # Reset the action spec of the agent to the train env action spec
        agent.egreedy_tdmodule._spec = train_env.action_spec  # noqa: SLF001

    return f


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/shim2018",
    config_name="config",
)
def main(cfg: Shim2018TrainConfig) -> None:
    # Evaluate alias arguments
    if cfg.hard_budget is not None:
        cfg.mdp.hard_budget = cfg.hard_budget

    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    log_fn: Callable[[dict[str, Any]], None]
    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cfg, job_type="training", tags=["shim2018"]
        )
        log_fn = run.log
    else:
        run = None
        log_fn = lambda _d: None  # noqa: E731

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.rl_training_loop.n_batches = 2

    # Prep: things we need to get before creating an environment
    train_dataset, val_dataset, initializer, unmasker, class_weights = (
        afa_rl_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    class_weights = class_weights.to(device)

    pretrained_model, pretrained_model_optim = get_shim2018_pretrained_model(
        pretrained_model_bundle_path=Path(cfg.pretrained_model_bundle_path),
        pretrained_model_lr=cfg.pretrained_model_lr,
        device=device,
    )

    train_env, eval_env = create_afa_envs(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        reward_fn=get_shim2018_reward_fn(
            pretrained_model=pretrained_model,
            weights=class_weights,
            acquisition_costs=(
                0 if cfg.soft_budget_param is None else cfg.soft_budget_param
            )
            * torch.ones(
                (
                    unmasker.get_n_selections(
                        feature_shape=train_dataset.feature_shape
                    ),
                ),
                device=class_weights.device,
            ),
            n_feature_dims=len(train_dataset.feature_shape),
        ),
        n_agents=cfg.mdp.n_agents,
        n_selections=unmasker.get_n_selections(
            feature_shape=train_dataset.feature_shape
        ),
        hard_budget=cfg.mdp.hard_budget,
        initialize_fn=initializer.initialize,
        unmask_fn=unmasker.unmask,
        force_hard_budget=cfg.mdp.force_hard_budget,
        device=device,
        seed=cfg.seed,
    )

    agent = Shim2018Agent(
        cfg=cfg.agent,
        embedder=pretrained_model.embedder,
        embedding_size=pretrained_model.embedder.encoder.output_size,
        action_spec=train_env.action_spec,
        action_mask_key="allowed_action_mask",
        module_device=device,
        n_feature_dims=len(train_dataset.feature_shape),
        n_batches=cfg.rl_training_loop.n_batches,
    )

    try:
        afa_rl_training_loop(
            cfg=cfg.rl_training_loop,
            train_env=train_env,
            eval_env=eval_env,
            agent=agent,
            post_process_batch_callback=get_post_process_batch_callback(
                pretrained_model=pretrained_model,
                pretrained_model_optim=pretrained_model_optim,
                activate_joint_training_after_batch=int(
                    cfg.rl_training_loop.n_batches
                    * cfg.activate_joint_training_after_fraction
                ),
                class_weights=class_weights,
                feature_shape=train_dataset.feature_shape,
            ),
            afa_predict_fn=Shim2018AFAPredictFn(pretrained_model),
            device=device,
            feature_shape=train_dataset.feature_shape,
            log_fn=log_fn,
            pre_eval_callback=get_pre_eval_callback(
                agent=agent, eval_env=eval_env
            ),
            post_eval_callback=get_post_eval_callback(
                agent=agent, train_env=train_env
            ),
        )
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    finally:
        log.info("Training completed, starting cleanup and model saving")
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_exploitative_policy().to("cpu"),
            Shim2018AFAClassifier(
                pretrained_model, device=torch.device("cpu")
            ),
        )

        save_bundle(
            obj=afa_method,
            path=Path(cfg.save_path),
            metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
        )

        if run is not None:
            run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log.info("Script completed successfully")


if __name__ == "__main__":
    main()
