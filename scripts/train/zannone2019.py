import gc
import torch
import wandb
import hydra
import logging
import matplotlib

from tqdm import tqdm
from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from torch.nn import functional as F
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type

from afabench.afa_rl.agents import Agent
from afabench.afa_rl.afa_env import AFAEnv
from afabench.afa_rl.utils import get_eval_metrics
from afabench.common.custom_types import AFADataset
from afabench.afa_rl.afa_methods import RLAFAMethod
from afabench.afa_rl.datasets import get_afa_dataset_fn
from afabench.afa_rl.zannone2019.agents import Zannone2019Agent
from afabench.common.config_classes import Zannone2019TrainConfig
from afabench.afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afabench.afa_rl.zannone2019.utils import load_pretrained_model_artifacts

from afabench.afa_rl.zannone2019.models import (
    Zannone2019AFAClassifier,
    Zannone2019AFAPredictFn,
    Zannone2019PretrainingModel,
)
from afabench.common.utils import (
    dict_with_prefix,
    get_class_probabilities,
    set_seed,
)


def visualize_digits(
    features: torch.Tensor, labels: torch.Tensor, shuffle: bool = True
):
    """Visualize 9 MNIST digits"""
    fig, axs = plt.subplots(3, 3)
    if shuffle:
        indices = torch.randperm(len(features))[:9]
    else:
        indices = torch.arange(9)
    for i in range(9):
        row = i // 3
        col = i % 3
        idx = indices[i]
        axs[row, col].imshow(features[idx].numpy().reshape((28, 28)))
        axs[row, col].set_title(labels[idx].argmax().item())
    return fig, axs


def visualize_pretrained_model(
    model: Zannone2019PretrainingModel,
    dataset: AFADataset,
    latent_size: int,
    device: torch.device,
    # dataset_type: str,
) -> None:
    indices = torch.randperm(len(dataset))[:5]
    features = dataset.features[indices].to(device)
    n_classes = dataset.labels.shape[-1]
    with_label_z, with_label_reconstructed_features = (
        model.fully_observed_reconstruction(
            features=features,
            n_classes=n_classes,
            label=dataset.labels[indices].to(device),
        )
    )
    with_label_z = with_label_z.cpu()
    with_label_reconstructed_features = with_label_reconstructed_features.cpu()
    without_label_z, without_label_reconstructed_features = (
        model.fully_observed_reconstruction(
            features=features,
            n_classes=n_classes,
            label=None,
        )
    )
    without_label_z = without_label_z.cpu()
    without_label_reconstructed_features = (
        without_label_reconstructed_features.cpu()
    )

    features = features.cpu()

    # Plot everything
    def _plot(
        _features: torch.Tensor,
        _z: torch.Tensor,
        _reconstructed_features: torch.Tensor,
    ):
        fig, axs = plt.subplots(5, 3)
        for i in range(5):
            axs[i, 0].plot(_features[i])
            axs[i, 0].set_title("Features")
            axs[i, 1].plot(_z[i])
            axs[i, 1].set_title("Latent vector")
            axs[i, 2].plot(_reconstructed_features[i])
            axs[i, 2].set_title("Reconstructed features")
        return fig, axs

    _with_label_fig, _with_label_axs = _plot(
        features, with_label_z, with_label_reconstructed_features
    )
    _with_label_fig.suptitle("With label")

    _without_label_fig, _without_label_axs = _plot(
        features, without_label_z, without_label_reconstructed_features
    )
    _without_label_fig.suptitle("Without label")

    _generation_fig, generation_axs = plt.subplots(5, 3)
    # Generate 5 latent vectors and reconstructed features
    generated_z, generated_features, generated_labels = model.generate_data(
        latent_size, device, n_samples=5
    )
    generated_z = generated_z.cpu()
    generated_features = generated_features.cpu()
    generated_labels = generated_labels.cpu()

    for i in range(5):
        generation_axs[i, 0].plot(generated_z[i])
        generation_axs[i, 0].set_title("Latent vector")
        generation_axs[i, 1].plot(generated_features[i])
        generation_axs[i, 1].set_title("Generated features")
        generation_axs[i, 2].plot(generated_labels[i])
        generation_axs[i, 2].set_title("Generated label")

    plt.show()

    # Visualize MNIST digits
    # fig_real, _axs_real = visualize_digits(
    #     train_dataset.features, train_dataset.labels, shuffle=False
    # )
    # fig_real.suptitle("Real digits")
    # if cfg.n_generated_samples >= 9:
    #     fig_fake, _axs_fake = visualize_digits(
    #         generated_features, generated_labels, shuffle=False
    #     )
    #     fig_fake.suptitle("Generated digits")
    # fig_label_recon, _axs_label_recon = visualize_digits(
    #     label_reconstructed_features, train_dataset.labels, shuffle=False
    # )
    # fig_label_recon.suptitle("Reconstructed digits using label")
    # fig_recon, _axs_recon = visualize_digits(
    #     reconstructed_features, train_dataset.labels, shuffle=False
    # )
    # fig_recon.suptitle("Reconstructed digits without using label")
    #
    # plt.show()
    #
    # Plot distribution of latent variables
    # plt.figure(figsize=(12, 4))
    # for i in range(min(z.shape[1], 10)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.hist(z[:, i], bins=30)
    #     plt.title(f"Latent dim {i}")
    # plt.tight_layout()
    # plt.show()


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019TrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["zannone2019"],
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
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.requires_grad_(
        False
    )  # zannone2019 does not train jointly

    if cfg.visualize:
        matplotlib.use("WebAgg")
        # Use pretrained model to reconstruct some samples. Visualize everything
        visualize_pretrained_model(
            pretrained_model,
            val_dataset,
            latent_size=pretrained_model_config.partial_vae.latent_size,
            device=device,
        )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )
    class_weights = 1 / train_class_probabilities
    class_weights = (class_weights / class_weights.sum()).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    reward_fn = get_zannone2019_reward_fn(
        pretrained_model=pretrained_model,
        weights=class_weights,
        acquisition_costs=torch.zeros(n_features, device=device)
        if cfg.cost_param is None
        else cfg.cost_param
        * train_dataset.get_feature_acquisition_costs().to(device),
    )

    if cfg.n_generated_samples > 0:
        # Use the pretrained model to generate new artificial data
        generated_features = torch.zeros(cfg.n_generated_samples, n_features)
        generated_labels = torch.zeros(cfg.n_generated_samples, n_classes)
        n_generation_batches = (
            cfg.n_generated_samples // cfg.generation_batch_size
        )
        for batch_idx in tqdm(
            range(n_generation_batches), desc="Generating artificial samples"
        ):
            _z, generated_features_batch, generated_labels_batch = (
                pretrained_model.generate_data(
                    latent_size=pretrained_model_config.partial_vae.latent_size,
                    device=device,
                    n_samples=cfg.generation_batch_size,
                )
            )
            generated_features[
                batch_idx * cfg.generation_batch_size : (batch_idx + 1)
                * cfg.generation_batch_size,
                :,
            ] = generated_features_batch.cpu()
            # Convert labels to one hot instead of continuous probabilities
            generated_labels[
                batch_idx * cfg.generation_batch_size : (batch_idx + 1)
                * cfg.generation_batch_size,
                :,
            ] = F.one_hot(
                generated_labels_batch.argmax(-1),
                num_classes=generated_labels_batch.shape[-1],
            ).cpu()

        train_features = generated_features
        train_labels = generated_labels
    else:
        train_features = train_dataset.features
        train_labels = train_dataset.labels

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(train_features, train_labels)
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

    agent: Agent = Zannone2019Agent(
        cfg=cfg.agent,
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        action_spec=train_env.action_spec,
        latent_size=pretrained_model_config.partial_vae.latent_size,
        action_mask_key="action_mask",
        batch_size=cfg.batch_size,
        module_device=device,
        replay_buffer_device=device,
    )

    collector = SyncDataCollector(
        train_env,
        agent.get_policy(),
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
        # device=device,
    )

    # Training loop
    try:
        for batch_idx, tds in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            # Collapse agent and batch dimensions
            td = tds.flatten(start_dim=0, end_dim=1)
            loss_info = agent.process_batch(td)

            # Log training info
            run.log(
                dict_with_prefix(
                    "train/",
                    loss_info
                    | dict_with_prefix("cheap_info.", agent.get_cheap_info())
                    | {
                        "reward": td["next", "reward"].mean().cpu().item(),
                        # "actions": wandb.Histogram(td["action"].cpu()),
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
                    td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, agent.get_exploitative_policy()
                        ).squeeze(0)
                        for _ in tqdm(
                            range(cfg.n_eval_episodes), desc="Evaluating"
                        )
                    ]
                metrics_eval = get_eval_metrics(
                    td_evals, Zannone2019AFAPredictFn(pretrained_model)
                )
                run.log(
                    dict_with_prefix(
                        "eval/",
                        dict_with_prefix("agent_policy.", metrics_eval)
                        # | dict_with_prefix("agent_train_policy.", train_metrics_eval)
                        | dict_with_prefix(
                            "expensive_info.", agent.get_expensive_info()
                        ),
                    )
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Convert the embedder+agent to an AFAMethod and save it as a temporary file
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_exploitative_policy().to("cpu"),
            Zannone2019AFAClassifier(
                pretrained_model,
                device=torch.device("cpu"),
            ),
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
                name=f"train_zannone2019-{
                    pretrained_model_config.dataset_artifact_name.split(':')[0]
                }-{budget_str}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "zannone2019",
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
