import gc
import logging
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.common.utils import (
    afa_discriminative_training_prep,
)
from afabench.components.methods.generative.eddi.afa_methods import (
    EDDIAFAMethod,
)
from afabench.components.methods.generative.eddi.config import (
    EDDITrainingConfig,
)
from afabench.components.methods.rl.odin.models import (
    ODINPretrainingModel,  # noqa: TC001
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.bundle_system.torch_bundle import (
    TorchModelBundle,  # noqa: TC001
)
from afabench.core.utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/eddi_builtin",
    config_name="config",
)
def main(cfg: EDDITrainingConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_dataset, _, _, unmasker, class_weights = (
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
    pretrained_model = cast("ODINPretrainingModel", torch_model_bundle.model)
    feature_costs = train_dataset.get_feature_acquisition_costs()
    selection_costs = unmasker.get_selection_costs(feature_costs)

    afa_method: EDDIAFAMethod = EDDIAFAMethod(
        sampler=pretrained_model.partial_vae,
        predictor=pretrained_model.classifier,
        num_classes=num_classes,
        classifier_bundle_path=None,
        selection_costs=selection_costs,
        n_contexts=getattr(unmasker, "n_contexts", None),
    )

    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"EDDI method saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
