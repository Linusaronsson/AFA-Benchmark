import gc
import logging
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.afa_generative.afa_methods import Ma2018AFAMethod
from afabench.common.bundle import load_bundle, save_bundle
from afabench.afa_discriminative.utils import afa_discriminative_training_prep
from afabench.afa_rl.zannone2019.models import Zannone2019PretrainingModel
from afabench.common.config_classes import Ma2018TrainingConfig
from afabench.common.torch_bundle import TorchModelBundle
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/ma2018",
    config_name="config",
)
def main(cfg: Ma2018TrainingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_dataset, _, _, _, class_weights = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    assert class_weights is not None
    class_weights = class_weights.to(device)
    num_classes = train_dataset.label_shape[-1]

    pretrained_model, _ = load_bundle(
        Path(cfg.pretrained_model_bundle_path),
        device=device,
    )
    torch_model_bundle = cast(
        "TorchModelBundle",
        cast("object", pretrained_model),
    )
    pretrained_model = cast(
        "Zannone2019PretrainingModel", torch_model_bundle.model
    )

    afa_method: Ma2018AFAMethod = Ma2018AFAMethod(
        sampler=pretrained_model.partial_vae,
        predictor=pretrained_model.classifier,
        num_classes=num_classes,
    )

    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"Ma2018 method saved to: {cfg.save_path}")

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
