import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from afabench.afa_rl.afa_methods import RLAFAMethod
from omegaconf.omegaconf import OmegaConf
from tensordict import TensorDict
from torch import optim
from torch.nn import functional as F
from torchrl.data import TensorSpec

from afabench.afa_rl.common.custom_types import AFARewardFn
from afabench.afa_rl.common.training import (
    afa_rl_training_loop,
    afa_rl_training_prep,
)

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
from afabench.afa_rl.shim2018.agents import Shim2018Agent
from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018AFAClassifier,
)

# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    Shim2018AgentConfig,
    Shim2018TrainConfig,
)
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from afabench.afa_rl.agent_interface import Agent

    from afabench.common.custom_types import AFADataset


log = logging.getLogger(__name__)


def post_process_batch_callback(td: TensorDict, batch_idx: int) -> None:
    # Train classifier and embedder jointly if we have reached the correct batch
    if batch_idx >= activate_joint_training_after_batch:
        if batch_idx == activate_joint_training_after_batch:
            log.info("Activating joint training of classifier and embedder")
        pretrained_model.train()
        pretrained_model_optim.zero_grad()

        _, logits_next = pretrained_model(
            td["next", "masked_features"], td["next", "feature_mask"]
        )
        # F.cross_entropy does not support multiple batch dimensions
        class_loss_next = F.cross_entropy(
            logits_next.flatten(end_dim=-2),
            td["next", "label"].flatten(end_dim=-2),
            weight=class_weights,
        )
        class_loss_next.mean().backward()

        pretrained_model_optim.step()
        pretrained_model.eval()
    else:
        class_loss_next = torch.zeros((1,), device=device, dtype=torch.float32)


def get_shim2018_pretrained_model_and_agent(
    pretrained_model_bundle_path: Path,
    pretrained_model_lr: float,
    train_dataset: AFADataset,
    agent_cfg: Shim2018AgentConfig,
    device: torch.device,
    action_spec: TensorSpec,
    frames_per_batch: int,
    n_batches: int,
) -> tuple[Shim2018Agent, LitShim2018EmbedderClassifier, optim.Adam]:
    """Construct everything particular to the shim2018 method, except for the reward function (compared to other RL methods)."""
    log.info("Loading pretrained model...")
    pretrained_model, _ = load_bundle(
        Path(pretrained_model_bundle_path),
        device=device,
    )
    pretrained_model = cast(
        "LitShim2018EmbedderClassifier",
        cast("object", pretrained_model.model),  # pyright: ignore[reportAttributeAccessIssue]
    )
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    pretrained_model_optim = optim.Adam(
        pretrained_model.parameters(), lr=pretrained_model_lr
    )
    log.info("Pretrained model loaded.")

    log.info("Creating Shim2018 agent")
    agent: Agent = Shim2018Agent(
        cfg=agent_cfg,
        embedder=pretrained_model.embedder,
        embedding_size=pretrained_model.embedder.encoder.output_size,
        action_spec=action_spec,
        action_mask_key="allowed_action_mask",
        batch_size=frames_per_batch,
        module_device=device,
        n_feature_dims=len(train_dataset.feature_shape),
        n_batches=n_batches,
    )
    log.info("Agent created successfully")

    return agent, pretrained_model, pretrained_model_optim


def get_shim2018_reward_fn() -> AFARewardFn:
    # Create reward function - temporarily using range-based for debugging
    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    # n_classes = len(train_class_probabilities)
    class_weights = 1 / train_class_probabilities
    class_weights = (class_weights / class_weights.sum()).to(device)
    n_selections = unmasker.get_n_selections(train_dataset.feature_shape)
    cost_per_selection = 0 if soft_budget_param is None else soft_budget_param
    reward_fn = get_shim2018_reward_fn(
        pretrained_model=pretrained_model,
        weights=class_weights,
        acquisition_costs=cost_per_selection
        * torch.ones(
            (n_selections,),
            device=device,
        ),
    )
    return reward_fn


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/shim2018",
    config_name="config",
)
def main(cfg: Shim2018TrainConfig) -> None:
    log.debug(cfg)
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
        cfg.afa_rl_training_loop.n_batches = 2

    train_dataset, val_dataset, initializer, unmasker = afa_rl_training_prep(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        initializer_cfg=cfg.initializer,
        unmasker_cfg=cfg.unmasker,
    )

    log.info("Constructing reward function...")
    reward_fn = get_shim2018_reward_fn()
    log.info("Reward function constructed.")

    # Create env

    agent, pretrained_model, pretrained_model_optim = (
        get_shim2018_pretrained_model_and_agent(
            pretrained_model_bundle_path=cfg.pretrained_model_bundle_path,
            pretrained_model_lr=cfg.pretrained_model_lr,
            train_dataset=train_dataset,
            agent_cfg=cfg.agent,
            device=device,
            action_spec=train_env.action_spec,
            frames_per_batch=cfg.afa_rl_training_loop.frames_per_batch,
            n_batches=cfg.afa_rl_training_loop.n_batches,
        )
    )

    activate_joint_training_after_batch = int(
        cfg.afa_rl_training_loop.n_batches
        * cfg.activate_joint_training_after_fraction
    )

    try:
        afa_rl_training_loop(
            cfg=cfg.afa_rl_training_loop,
            train_env=train_env,
            eval_env=eval_env,
            agent=agent,
            post_process_batch_callback=post_process_batch_callback,
            train_log_fn=train_log_fn,
            afa_predict_fn=afa_predict_fn,
            log_fn=log_fn,
            pre_eval_callback=pre_eval_callback,
            post_eval_callback=post_eval_callback,
        )
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    finally:
        log.info("Training completed, starting cleanup and model saving")
        log.info("Converting model to CPU and creating AFA method...")
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_exploitative_policy().to("cpu"),
            Shim2018AFAClassifier(
                pretrained_model, device=torch.device("cpu")
            ),
        )
        log.info("AFA method created.")

        log.info("Saving method to local filesystem...")
        save_bundle(
            obj=afa_method,
            path=Path(cfg.save_path),
            metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
        )
        log.info("Saved trained method successfully.")

        if run is not None:
            run.finish()

        log.info("Running garbage collection and clearing CUDA cache")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log.info("Script completed successfully")


if __name__ == "__main__":
    main()
