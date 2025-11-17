import gc
import wandb
import torch
import hydra
import logging

from tqdm import tqdm
from pathlib import Path
from typing import Any, cast
from dacite import from_dict
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type

from afabench.afa_rl.agents import Agent
from afabench.afa_rl.afa_env import AFAEnv
from afabench.common.custom_types import AFADataset
from afabench.afa_rl.afa_methods import RLAFAMethod
from afabench.afa_rl.datasets import get_afa_dataset_fn
from afabench.afa_rl.kachuee2019.agents import Kachuee2019Agent
from afabench.afa_rl.kachuee2019.reward import get_kachuee2019_reward_fn
from afabench.afa_rl.kachuee2019.utils import get_kachuee2019_model_from_config

from afabench.afa_rl.kachuee2019.models import (
    Kachuee2019AFAClassifier,
    Kachuee2019AFAPredictFn,
    LitKachuee2019PQModule,
)
from afabench.afa_rl.utils import (
    get_eval_metrics,
)
from afabench.common.config_classes import (
    Kachuee2019PretrainConfig,
    Kachuee2019TrainConfig,
)
from afabench.common.utils import (
    dict_with_prefix,
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)


def load_pretrained_model_artifacts(
    artifact_name: str,
) -> tuple[
    AFADataset,  # train dataset
    AFADataset,  # val dataset
    AFADataset,  # test dataset
    dict[str, Any],  # dataset metadata
    LitKachuee2019PQModule,
    Kachuee2019PretrainConfig,
]:
    """Load a pretrained model and the dataset it was trained on, from a WandB artifact."""
    pretrained_model_artifact = wandb.use_artifact(
        artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    # The dataset dir should contain a file called model.pt
    artifact_filenames = [
        f.name for f in pretrained_model_artifact_dir.iterdir()
    ]
    assert {"model.pt"}.issubset(
        artifact_filenames
    ), f"Dataset artifact must contain a model.pt file. Instead found: {
        artifact_filenames
    }"

    # Access config of the run that produced this pretrained model
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Kachuee2019PretrainConfig = from_dict(
        data_class=Kachuee2019PretrainConfig, data=pretrained_model_config_dict
    )

    # Load the dataset that the pretrained model was trained on
    train_dataset, val_dataset, test_dataset, dataset_metadata = (
        load_dataset_artifact(pretrained_model_config.dataset_artifact_name)
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )

    pretrained_model = get_kachuee2019_model_from_config(
        pretrained_model_config,
        n_features,
        n_classes,
        train_class_probabilities,
    )
    pretrained_model_checkpoint = torch.load(
        pretrained_model_artifact_dir / "model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/kachuee2019",
    config_name="config",
)
def main(cfg: Kachuee2019TrainConfig):  # noqa: PLR0915
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["kachuee2019"],
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

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
    assert not (cfg.hard_budget is not None and cfg.cost_param is not None), (
        "Only one of hard_budget or cost_param can be specified, not both."
    )

    # Load pretrained model and dataset from artifacts
    (
        train_dataset,
        val_dataset,
        _,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    ) = load_pretrained_model_artifacts(cfg.pretrained_model_artifact_name)
    pq_module = pretrained_model.pq_module.to(device)

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )
    class_weights = 1 / train_class_probabilities
    class_weights = (class_weights / class_weights.sum()).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    reward_fn = get_kachuee2019_reward_fn(
        pq_module=pq_module,
        method=cfg.reward_method,
        mcdrop_samples=cfg.mcdrop_samples,
        acquisition_costs=torch.zeros(n_features, device=device)
        if cfg.cost_param is None
        else cfg.cost_param
        * train_dataset.get_feature_acquisition_costs().to(device),
    )

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(
        train_dataset.features, train_dataset.labels
    )
    val_dataset_fn = get_afa_dataset_fn(
        val_dataset.features, val_dataset.labels
    )

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

    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )

    agent: Agent = Kachuee2019Agent(
        action_spec=train_env.action_spec,
        action_mask_key="action_mask",
        module_device=torch.device(cfg.device),
        replay_buffer_device=torch.device(cfg.device),
        pq_module=pq_module,
        class_weights=class_weights,
        cfg=cfg.agent,
    )

    collector = SyncDataCollector(
        train_env,
        agent.get_policy(),
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
    )
    # Training loop
    try:
        for batch_idx, tds in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            # Collapse agent and batch dimensions
            td = tds.flatten(start_dim=0, end_dim=1)
            process_batch_info = agent.process_batch(td)

            # Log training info
            run.log(
                dict_with_prefix(
                    "train/",
                    dict_with_prefix("process_batch.", process_batch_info)
                    | dict_with_prefix("cheap_info.", agent.get_cheap_info())
                    | {
                        "reward": td["next", "reward"].mean().cpu().item(),
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
                    },
                )
            )

            if (
                batch_idx != 0
                and cfg.eval_every_n_batches is not None
                and batch_idx % cfg.eval_every_n_batches == 0
            ):
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    # HACK: Set the action spec of the agent to the eval env action spec
                    agent.egreedy_tdmodule._spec = eval_env.action_spec  # pyright: ignore
                    td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, agent.get_exploitative_policy()
                        ).squeeze(0)
                        for _ in tqdm(
                            range(cfg.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_tdmodule._spec = train_env.action_spec  # pyright: ignore
                metrics_eval = get_eval_metrics(
                    td_evals, Kachuee2019AFAPredictFn(pq_module)
                )
                run.log(
                    dict_with_prefix(
                        "eval/",
                        metrics_eval
                        | dict_with_prefix(
                            "expensive_info.", agent.get_expensive_info()
                        ),
                    )
                )

    except KeyboardInterrupt:
        pass
    finally:
        afa_method = RLAFAMethod(
            agent.get_policy().to("cpu"),
            Kachuee2019AFAClassifier(pq_module, device=torch.device("cpu")),
            acquisition_cost=cfg.cost_param,
        )
        # Save the method to a temporary directory and load it again to ensure it is saved correctly
        with TemporaryDirectory(delete=False) as tmp_path_str:
            tmp_path = Path(tmp_path_str)
            afa_method.save(tmp_path)
            del afa_method

            # Save the model as a WandB artifact
            log.info("Creating WandB artifact for trained method")
            if cfg.cost_param is not None:
                budget_str = f"costparam_{cfg.cost_param}"
            else:
                budget_str = f"budget_{cfg.hard_budget}"
            afa_method_artifact = wandb.Artifact(
                name=f"train_kachuee2019-{
                    pretrained_model_config.dataset_artifact_name.split(':')[0]
                }-{budget_str}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "kachuee2019",
                    "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                    "dataset_type": dataset_metadata["dataset_type"],
                    "budget": cfg.hard_budget,
                    "seed": cfg.seed,
                },
            )

            afa_method_artifact.add_dir(str(tmp_path))
            run.log_artifact(
                afa_method_artifact, aliases=cfg.output_artifact_aliases
            )

        run.finish()

        gc.collect()  # Force Python GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
            torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
