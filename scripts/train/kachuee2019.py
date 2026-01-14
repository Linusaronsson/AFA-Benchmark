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
from afabench.afa_rl.kachuee2019.agents import Kachuee2019Agent
from afabench.afa_rl.kachuee2019.models import (
    Kachuee2019AFAClassifier,
    Kachuee2019AFAPredictFn,
    Kachuee2019PQModule,
    LitKachuee2019PQModule,
)
from afabench.afa_rl.kachuee2019.reward import get_kachuee2019_reward_fn
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    Kachuee2019TrainConfig,
)
from afabench.common.utils import (
    initialize_wandb_run,
    set_seed,
)

if TYPE_CHECKING:
    from afabench.common.torch_bundle import TorchModelBundle

log = logging.getLogger(__name__)


def get_post_process_batch_callback(
    pq_module: Kachuee2019PQModule,
    pq_module_optim: optim.Adam,
    activate_joint_training_after_batch: int,
    class_weights: torch.Tensor,
    feature_shape: torch.Size,
) -> Callable[[TensorDict, int], dict[str, Any]]:
    def f(td: TensorDict, batch_idx: int) -> dict[str, Any]:
        # Train classifier and embedder jointly if we have reached the correct batch
        assert td.batch_dims == 2, "Expected two batch dimensions"

        if batch_idx >= activate_joint_training_after_batch:
            if batch_idx == activate_joint_training_after_batch:
                log.info("Activating joint training of classifier")
            pq_module.train()
            pq_module_optim.zero_grad()

            n_feature_dims = len(feature_shape)

            # Flatten feature dims
            flat_masked_features = td["next", "masked_features"].flatten(
                start_dim=-n_feature_dims
            )
            assert flat_masked_features.ndim == td["next", "label"].ndim, (
                "Label should be 1D"
            )

            # Flatten batch dims
            flat_masked_features = flat_masked_features.flatten(end_dim=-2)
            flat_label = td["next", "label"].flatten(end_dim=-2)

            logits_next, _qvalues = pq_module.forward(flat_masked_features)
            class_loss_next = F.cross_entropy(
                logits_next,
                flat_label,
                weight=class_weights,
            )
            class_loss_next.mean().backward()

            pq_module_optim.step()
            pq_module.eval()

            return {"avg_class_loss": class_loss_next.mean().cpu().item()}
        return {}

    return f


def get_kachuee2019_pretrained_model(
    pretrained_model_bundle_path: Path,
    pretrained_model_lr: float,
    device: torch.device,
) -> tuple[LitKachuee2019PQModule, optim.Adam]:
    pretrained_model, _ = load_bundle(
        Path(pretrained_model_bundle_path),
        device=device,
    )
    torch_model_bundle = cast(
        "TorchModelBundle",
        cast("object", pretrained_model),
    )
    pretrained_model = cast("LitKachuee2019PQModule", torch_model_bundle.model)
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    pretrained_model_optim = optim.Adam(
        pretrained_model.parameters(), lr=pretrained_model_lr
    )
    return pretrained_model, pretrained_model_optim


def get_pre_eval_callback(
    agent: Kachuee2019Agent, eval_env: AFAEnv
) -> Callable[[], None]:
    def f() -> None:
        # HACK: Set the action spec of the agent to the eval env action spec
        agent.egreedy_tdmodule._spec = eval_env.action_spec  # noqa: SLF001

    return f


def get_post_eval_callback(
    agent: Kachuee2019Agent, train_env: AFAEnv
) -> Callable[[], None]:
    def f() -> None:
        # Reset the action spec of the agent to the train env action spec
        agent.egreedy_tdmodule._spec = train_env.action_spec  # noqa: SLF001

    return f


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/kachuee2019",
    config_name="config",
)
def main(cfg: Kachuee2019TrainConfig) -> None:
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
    replay_buffer_device = (
        device
        if cfg.replay_buffer_device_same_as_device
        else torch.device("cpu")
    )

    log_fn: Callable[[dict[str, Any]], None]
    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cfg, job_type="training", tags=["kachuee2019"]
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

    pretrained_model, pretrained_model_optim = (
        get_kachuee2019_pretrained_model(
            pretrained_model_bundle_path=Path(
                cfg.pretrained_model_bundle_path
            ),
            pretrained_model_lr=cfg.pretrained_model_lr,
            device=device,
        )
    )

    train_env, eval_env = create_afa_envs(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        reward_fn=get_kachuee2019_reward_fn(
            pretrained_model=pretrained_model.pq_module,
            selection_costs=(
                0 if cfg.soft_budget_param is None else cfg.soft_budget_param
            )
            * unmasker.get_selection_costs(
                feature_costs=train_dataset.get_feature_acquisition_costs()
            ).to(device),
            n_feature_dims=len(train_dataset.feature_shape),
            method=cfg.reward_method,
            mcdrop_samples=cfg.mcdrop_samples,
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

    agent = Kachuee2019Agent(
        cfg=cfg.agent,
        pq_module=pretrained_model.pq_module,
        action_spec=train_env.action_spec,
        action_mask_key="allowed_action_mask",
        module_device=device,
        replay_buffer_device=replay_buffer_device,
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
                pq_module=pretrained_model.pq_module,
                pq_module_optim=pretrained_model_optim,
                activate_joint_training_after_batch=int(
                    cfg.rl_training_loop.n_batches
                    * cfg.activate_joint_training_after_fraction
                ),
                class_weights=class_weights,
                feature_shape=train_dataset.feature_shape,
            ),
            afa_predict_fn=Kachuee2019AFAPredictFn(pretrained_model.pq_module),
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
            Kachuee2019AFAClassifier(
                pretrained_model.pq_module, device=torch.device("cpu")
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
