import gc
import torch
import wandb
import hydra
import logging

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory

from afabench.common.utils import set_seed
from afabench.afa_generative.afa_methods import Ma2018AFAMethod

from afabench.afa_rl.zannone2019.utils import (
    load_pretrained_model_artifacts,
)
from afabench.common.config_classes import (
    Ma2018TrainingConfig,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/ma2018",
    config_name="config",
)
def main(cfg: Ma2018TrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["EDDI"],
        dir="extra/wandb",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    (
        train_dataset,
        val_dataset,
        _,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    ) = load_pretrained_model_artifacts(cfg.pretrained_model_artifact_name)
    num_classes = train_dataset.labels.shape[-1]
    afa_method: Ma2018AFAMethod = Ma2018AFAMethod(
        sampler=pretrained_model.partial_vae,
        predictor=pretrained_model.classifier,
        num_classes=num_classes,
    )

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)
        del afa_method
        afa_method = Ma2018AFAMethod.load(tmp_path, device=device)
        afa_method_artifact = wandb.Artifact(
            name=f"train_ma2018-{
                pretrained_model_config.dataset_artifact_name.split(':')[0]
            }-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "ma2018",
                "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        afa_method_artifact.add_file(str(tmp_path / "model.pt"))
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
