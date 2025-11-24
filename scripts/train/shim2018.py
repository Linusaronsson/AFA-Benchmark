import gc
import torch
import wandb
import hydra
import logging

from tqdm import tqdm
from torch import optim
from pathlib import Path
from dacite import from_dict
from omegaconf import OmegaConf
from torch.nn import functional as F
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type

from afabench import SAVE_PATH
from afabench.afa_rl.afa_env import AFAEnv
from afabench.afa_rl.afa_methods import RLAFAMethod
from afabench.afa_rl.datasets import get_afa_dataset_fn
from afabench.afa_rl.shim2018.agents import Shim2018Agent
from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.afa_rl.shim2018.utils import get_shim2018_model_from_config
from afabench.common.config_classes import (
    Shim2018PretrainConfig,
    Shim2018TrainConfig,
)

from afabench.afa_rl.shim2018.models import (
    Shim2018AFAClassifier,
    Shim2018AFAPredictFn,
)
from afabench.afa_rl.utils import (
    get_eval_metrics,
    module_norm,
)
from afabench.common.utils import (
    dict_with_prefix,
    get_class_probabilities,
    load_pretrained_model,
    set_seed,
)

if TYPE_CHECKING:
    from afabench.afa_rl.agents import Agent


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/shim2018",
    config_name="config",
)
def main(cfg: Shim2018TrainConfig) -> None:  # noqa: PLR0915
    log.info("Starting Shim2018 training script")
    log.debug(cfg)
    if cfg.seed is None:
        log.info("No seed specified, using random seed")
    else:
        log.info(f"Setting seed to {cfg.seed}")
    actual_seed = set_seed(cfg.seed)
    log.info(f"Using seed: {actual_seed}")
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Two possible cases: hard budget or soft budget
    if cfg.hard_budget is None:
        assert cfg.cost_param is not None, (
            "If no hard budget is specified, a cost_param must be given for soft budget training."
        )
        log.info("Detected soft budget case")
    if cfg.cost_param is None:
        assert cfg.hard_budget is not None, (
            "If no cost_param is specified, a hard budget must be given for hard budget training."
        )
        log.info("Detected hard budget case")
    assert not (
        cfg.hard_budget is not None and cfg.cost_param is not None
    ), f"Only one of hard_budget or cost_param can be specified, not both. hard_budget detected as {
        cfg.hard_budget
    } (type: {type(cfg.hard_budget)}), cost_param detected as {
        cfg.cost_param
    } (type: {type(cfg.cost_param)})"

    log.info("Initializing Weights & Biases run")
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["shim2018"],
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load pretrained model and dataset from artifacts
    log.info(
        f"Loading pretrained model from artifact: {
            cfg.pretrained_model_artifact_name
        }"
    )
    (
        pretrained_ckpt_path,
        metadata,
        pretrain_cfg,
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
    ) = load_pretrained_model(
        f"{cfg.pretrained_model_artifact_name}_seed_{cfg.seed}",
        device=device,
    )

    # Convert pretrain config dict to dataclass
    pretrained_model_config = from_dict(
        data_class=Shim2018PretrainConfig, data=pretrain_cfg
    )

    # Get dimensions
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )
    class_weights = 1 / train_class_probabilities
    class_weights = (class_weights / class_weights.sum()).to(device)

    # Instantiate and load pretrained model
    pretrained_model = get_shim2018_model_from_config(
        pretrained_model_config,
        n_features,
        n_classes,
        train_class_probabilities,
    )
    checkpoint = torch.load(pretrained_ckpt_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint["state_dict"])

    log.info("Successfully loaded pretrained model and datasets")

    log.info("Setting up pretrained model")
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)

    pretrained_model_optim = optim.Adam(
        pretrained_model.parameters(), lr=cfg.pretrained_model_lr
    )
    log.info("Pretrained model setup complete")

    log.info("Creating reward function")
    reward_fn = get_shim2018_reward_fn(
        pretrained_model=pretrained_model,
        weights=class_weights,
        acquisition_costs=torch.zeros(n_features, device=device)
        if cfg.cost_param is None
        else cfg.cost_param
        * train_dataset.get_feature_acquisition_costs().to(device),
    )
    log.info("Reward function created")

    # MDP expects special dataset functions
    log.info("Creating dataset functions for environments")
    train_dataset_fn = get_afa_dataset_fn(
        train_dataset.features, train_dataset.labels, device=device
    )
    val_dataset_fn = get_afa_dataset_fn(
        val_dataset.features, val_dataset.labels, device=device
    )
    log.info("Dataset functions created")

    log.info("Creating training environment")
    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )
    check_env_specs(train_env)
    log.info("Training environment created and validated")

    log.info("Creating evaluation environment")
    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )
    log.info("Evaluation environment created")

    log.info("Creating Shim2018 agent")
    agent: Agent = Shim2018Agent(
        cfg=cfg.agent,
        embedder=pretrained_model.embedder,
        embedding_size=pretrained_model_config.encoder.output_size,
        action_spec=train_env.action_spec,
        action_mask_key="action_mask",
        batch_size=cfg.batch_size,
        module_device=torch.device(cfg.device),
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
        for batch_idx, tds in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            # Collapse agent and batch dimensions
            td = tds.flatten(start_dim=0, end_dim=1)
            loss_info = agent.process_batch(td)

            # Train classifier and embedder jointly if we have reached the correct batch
            if batch_idx >= cfg.activate_joint_training_after_n_batches:
                if batch_idx == cfg.activate_joint_training_after_n_batches:
                    log.info(
                        "Activating joint training of classifier and embedder"
                    )
                pretrained_model.train()
                pretrained_model_optim.zero_grad()

                _, logits_next = pretrained_model(
                    td["next", "masked_features"], td["next", "feature_mask"]
                )
                class_loss_next = F.cross_entropy(
                    logits_next, td["next", "label"], weight=class_weights
                )
                class_loss_next.mean().backward()

                pretrained_model_optim.step()
                pretrained_model.eval()
            else:
                class_loss_next = torch.zeros(
                    (1,), device=device, dtype=torch.float32
                )

            # Log training info
            run.log(
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
                        "avg stop time": td["feature_mask"][td["action"] == 0]
                        .sum(-1)
                        .float()
                        .mean()
                        .cpu()
                        .item(),
                        "batch_idx": batch_idx,
                    }
                    | {"class_loss": class_loss_next.mean().cpu().item()},
                )
            )

            if (
                batch_idx != 0
                and cfg.eval_every_n_batches is not None
                and batch_idx % cfg.eval_every_n_batches == 0
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
                run.log(
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
        log.info("Converting model to CPU and creating AFA method")
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_exploitative_policy().to("cpu"),
            Shim2018AFAClassifier(
                pretrained_model, device=torch.device("cpu")
            ),
            cfg.cost_param,
        )
        log.info("AFA method created")

        # Save locally
        log.info("Saving method to local filesystem")
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            afa_method.save(tmp_path)

            if cfg.cost_param is not None:
                budget_str = f"costparam_{cfg.cost_param}"
            else:
                budget_str = f"budget_{cfg.hard_budget}"

            split = dataset_metadata["split_idx"]
            dataset_type = dataset_metadata["dataset_type"]

            artifact_identifier = f"{dataset_type.lower()}_split_{split}_{
                budget_str
            }_seed_{cfg.seed}"
            artifact_dir = SAVE_PATH / artifact_identifier

            metadata_out = {
                "method_type": "RLAFAMethod",
                "dataset_type": dataset_type,
                "dataset_artifact_name": metadata["dataset_artifact_name"],
                "budget": cfg.hard_budget
                if cfg.hard_budget is not None
                else None,
                "cost_param": cfg.cost_param
                if cfg.cost_param is not None
                else None,
                "seed": cfg.seed,
                "split_idx": split,
            }

            save_artifact(
                artifact_dir=artifact_dir,
                files={f.name: f for f in tmp_path.iterdir() if f.is_file()},
                metadata=metadata_out,
            )

            log.info(f"Shim2018 method saved to: {artifact_dir}")

        log.info("Finishing WandB run")
        run.finish()

        log.info("Running garbage collection and clearing CUDA cache")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log.info("Script completed successfully")


if __name__ == "__main__":
    main()
