import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf.omegaconf import OmegaConf
from torch.nn import functional as F

from afabench.afa_rl.common.afa_methods import RLAFAMethod
from afabench.afa_rl.common.training import (
    afa_rl_training_loop,
    afa_rl_training_prep,
    create_afa_envs,
)

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
from afabench.afa_rl.zannone2019.agents import Zannone2019Agent
from afabench.afa_rl.zannone2019.models import (
    Zannone2019AFAClassifier,
    Zannone2019AFAPredictFn,
    Zannone2019PretrainingModel,
)

# from afabench.afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afabench.afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    Zannone2019TrainConfig,
)
from afabench.common.custom_types import Features, Label
from afabench.common.datasets.wrappers import ExtendedAFADataset
from afabench.common.utils import (
    initialize_wandb_run,
    set_seed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from afabench.common.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def get_zannone2019_pretrained_model(
    pretrained_model_bundle_path: Path,
    device: torch.device,
) -> Zannone2019PretrainingModel:
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


def generate_data_batched(
    pretrained_model: Zannone2019PretrainingModel,
    samples: int,
    batch_size: int,
) -> tuple[Features, Label]:
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


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019TrainConfig) -> None:
    # Evaluate alias arguments
    # Flat hard budget parameter always overrides
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
            cfg=cfg, job_type="training", tags=["zannone2019"]
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

    pretrained_model = get_zannone2019_pretrained_model(
        pretrained_model_bundle_path=Path(cfg.pretrained_model_bundle_path),
        device=device,
    )

    # zannone2019 unique step: generate additional data using generative model
    if cfg.n_generated_samples > 0:
        additional_features, additional_labels = generate_data_batched(
            pretrained_model=pretrained_model,
            samples=cfg.n_generated_samples,
            batch_size=cfg.generation_batch_size,
        )
        extended_train_dataset = ExtendedAFADataset(
            base_dataset=train_dataset,
            additional_features=additional_features,
            additional_labels=additional_labels,
        )
    else:
        extended_train_dataset = train_dataset

    train_env, eval_env = create_afa_envs(
        train_dataset=extended_train_dataset,
        val_dataset=val_dataset,
        reward_fn=get_zannone2019_reward_fn(
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

    agent = Zannone2019Agent(
        cfg=cfg.agent,
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        action_spec=train_env.action_spec,
        latent_size=pretrained_model.latent_size,
        action_mask_key="allowed_action_mask",
        frames_per_batch=cfg.rl_training_loop.frames_per_batch,
        module_device=device,
        n_feature_dims=len(extended_train_dataset.feature_shape),
    )

    try:
        afa_rl_training_loop(
            cfg=cfg.rl_training_loop,
            train_env=train_env,
            eval_env=eval_env,
            agent=agent,
            post_process_batch_callback=None,
            afa_predict_fn=Zannone2019AFAPredictFn(pretrained_model),
            device=device,
            feature_shape=extended_train_dataset.feature_shape,
            log_fn=log_fn,
            pre_eval_callback=None,
            post_eval_callback=None,
        )
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    finally:
        log.info("Training completed, starting cleanup and model saving")
        log.info("Converting model to CPU and creating AFA method...")
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_exploitative_policy().to("cpu"),
            Zannone2019AFAClassifier(
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
