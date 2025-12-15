import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from rl_helpers import dict_with_prefix
from torch import optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from tqdm import tqdm

from afabench.afa_rl.afa_env import AFAEnv
from afabench.afa_rl.afa_methods import RLAFAMethod
from afabench.afa_rl.datasets import get_afa_dataset_fn

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
from afabench.afa_rl.shim2018.agents import Shim2018Agent
from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018AFAClassifier,
    Shim2018AFAPredictFn,
)

# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.afa_rl.utils import (
    get_eval_metrics,
    module_norm,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    Shim2018TrainConfig,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from afabench.afa_rl.agents import Agent
    from afabench.common.custom_types import AFADataset


log = logging.getLogger(__name__)


def should_evaluate_at_batch(
    batch_idx: int, n_batches: int, eval_n_times: int | None
) -> bool:
    """
    Determine if evaluation should be performed at the current batch.

    Args:
        batch_idx: Current batch index (0-based)
        n_batches: Total number of batches in training
        eval_n_times: Total number of evaluations desired across training

    Returns:
        True if evaluation should be performed at this batch, False otherwise
    """
    if eval_n_times is None or eval_n_times <= 0 or batch_idx == 0:
        return False

    eval_interval = n_batches // eval_n_times
    return eval_interval > 0 and batch_idx % eval_interval == 0


# def load_pretrained_model_artifact(
#     path: Path,
# ) -> LitShim2018EmbedderClassifier:
#     pass


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/shim2018",
    config_name="config",
)
def main(cfg: Shim2018TrainConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
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
        cfg.n_batches = 2

    log.info("Loading datasets...")
    train_dataset, _train_dataset_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    train_features, train_labels = train_dataset.get_all_data()
    val_dataset, _val_dataset_manifest = load_bundle(
        Path(cfg.val_dataset_bundle_path),
    )
    val_dataset = cast("AFADataset", cast("object", val_dataset))
    val_features, val_labels = val_dataset.get_all_data()
    log.info("Datasets loaded.")

    log.info("Loading pretrained model...")
    pretrained_model, _ = load_bundle(
        Path(cfg.pretrained_model_bundle_path),
        device=device,
    )
    pretrained_model = cast(
        "LitShim2018EmbedderClassifier",
        cast("object", pretrained_model.model),  # pyright: ignore[reportAttributeAccessIssue]
    )
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    pretrained_model_optim = optim.Adam(
        pretrained_model.parameters(), lr=cfg.pretrained_model_lr
    )
    log.info("Pretrained model loaded.")

    # Create initializer
    log.info("Creating initializer...")
    initializer = get_afa_initializer_from_config(cfg.initializer)
    log.info("Initializer created.")

    # Create unmasker
    log.info("Creating unmasker...")
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
    log.info("Unmasker created.")

    # Create reward function - temporarily using range-based for debugging
    log.info("Constructing reward function...")
    train_class_probabilities = get_class_frequencies(train_labels)
    n_classes = len(train_class_probabilities)
    class_weights = 1 / train_class_probabilities
    class_weights = (class_weights / class_weights.sum()).to(device)
    n_selections = unmasker.get_n_selections(train_dataset.feature_shape)
    cost_per_selection = (
        0 if cfg.soft_budget_param is None else cfg.soft_budget_param
    )

    # reward_fn = get_range_based_reward_fn(
    #     reward_ranges=[(5, 9)], reward_value=1.0
    # )
    reward_fn = get_shim2018_reward_fn(
        pretrained_model=pretrained_model,
        weights=class_weights,
        acquisition_costs=cost_per_selection
        * torch.ones(
            (n_selections,),
            device=device,
        ),
    )
    log.info("Reward function constructed (using range-based for debugging).")

    # MDP expects special dataset functions
    log.info("Creating dataset functions for environments...")
    train_dataset_fn = get_afa_dataset_fn(
        train_features, train_labels, device=device
    )
    val_dataset_fn = get_afa_dataset_fn(
        val_features, val_labels, device=device
    )
    log.info("Dataset functions created.")

    log.info("Creating training environment")
    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_shape=train_dataset.feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
        initialize_fn=initializer.initialize,
        unmask_fn=unmasker.unmask,
        force_hard_budget=cfg.force_hard_budget,
        seed=cfg.seed,
    )
    check_env_specs(train_env)
    log.info("Training environment created and validated")

    log.info("Creating evaluation environment")
    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_shape=val_dataset.feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
        initialize_fn=initializer.initialize,
        unmask_fn=unmasker.unmask,
        force_hard_budget=cfg.force_hard_budget,
        seed=cfg.seed,
    )
    log.info("Evaluation environment created")

    log.info("Creating Shim2018 agent")
    agent: Agent = Shim2018Agent(
        cfg=cfg.agent,
        embedder=pretrained_model.embedder,
        embedding_size=pretrained_model.embedder.encoder.output_size,
        action_spec=train_env.action_spec,
        action_mask_key="allowed_action_mask",
        batch_size=cfg.batch_size,
        module_device=torch.device(cfg.device),
        n_feature_dims=len(train_dataset.feature_shape),
        n_batches=cfg.n_batches,
    )
    log.info("Agent created successfully")

    log.info("Creating data collector")
    collector = SyncDataCollector(
        train_env,
        agent.get_exploratory_policy(),
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
        device=device,
    )
    log.info("Data collector created")

    # Training loop
    log.info(f"Starting training loop for {cfg.n_batches} batches")
    try:
        for batch_idx, td in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            if batch_idx == 200:
                pass

            # Collapse agent and batch dimensions
            loss_info = agent.process_batch(td)

            # Train classifier and embedder jointly if we have reached the correct batch
            activate_joint_training_after_batch = int(
                cfg.n_batches * cfg.activate_joint_training_after_fraction
            )
            if batch_idx >= activate_joint_training_after_batch:
                if batch_idx == activate_joint_training_after_batch:
                    log.info(
                        "Activating joint training of classifier and embedder"
                    )
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
                class_loss_next = torch.zeros(
                    (1,), device=device, dtype=torch.float32
                )

            # Log training info
            log_fn(
                dict_with_prefix(
                    "train/",
                    loss_info
                    | dict_with_prefix("cheap_info.", agent.get_cheap_info())
                    | {
                        "reward": td["next", "reward"].mean().item(),
                        # "action value": td["action_value"].mean().item(),
                        "chosen action value": td["chosen_action_value"]
                        .mean()
                        .cpu()
                        .item(),
                        # Average number of features selected when we stop
                        "fraction observed at stop time": td[
                            "next", "feature_mask"
                        ][td["next", "done"].squeeze(-1)]
                        .float()
                        .mean()
                        .cpu()
                        .item(),
                        # Action histogram, comment if too expensive
                        # "actions": wandb.Histogram(
                        #     td["action"].flatten().cpu().numpy()
                        # ),
                        # "cube_correct_action_ratio": (td["action"] <= 11)
                        # .float()
                        # .mean()
                        # .item(),
                        # "synthetic_mnist_correct_action_ratio": (
                        #     right_side_ratio_from_mask(
                        #         td["next", "feature_mask"][
                        #             td["next", "done"].squeeze(-1)
                        #         ]
                        #     ).mean()
                        # )
                        # .float()
                        # .mean()
                        # .item(),
                        # "action_value_distribution": wandb.Histogram(
                        #     td["action_value"][td["action_value"] > -100]
                        #     .flatten()
                        #     .cpu()
                        # ),
                        "batch_idx": batch_idx,
                    }
                    | {"class_loss": class_loss_next.mean().cpu().item()},
                )
            )

            if should_evaluate_at_batch(
                batch_idx, cfg.n_batches, cfg.eval_n_times
            ):
                log.info(f"Running evaluation at batch {batch_idx}")
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    # HACK: Set the action spec of the agent to the eval env action spec
                    agent.egreedy_tdmodule._spec = eval_env.action_spec  # noqa: SLF001
                    td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, agent.get_exploitative_policy()
                        ).squeeze(0)
                        for _ in tqdm(
                            range(cfg.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_tdmodule._spec = train_env.action_spec  # noqa: SLF001
                    metrics_eval = get_eval_metrics(
                        td_evals, Shim2018AFAPredictFn(pretrained_model)
                    )
                log_fn(
                    dict_with_prefix(
                        "eval/",
                        dict_with_prefix("agent_policy.", metrics_eval)
                        | dict_with_prefix(
                            "expensive_info.", agent.get_expensive_info()
                        )
                        | {
                            "classifier_norm": module_norm(
                                pretrained_model.classifier
                            ),
                            "embedder_norm": module_norm(
                                pretrained_model.embedder
                            ),
                        },
                    )
                )
                log.info(f"Evaluation completed at batch {batch_idx}")

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
