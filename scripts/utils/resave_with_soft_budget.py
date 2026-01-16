import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import ResaveConfig
from afabench.common.custom_types import AFAMethod  # noqa: TC001

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/utils",
    config_name="config",
)
def main(cfg: ResaveConfig) -> None:
    device = torch.device(cfg.device)

    afa_method, manifest = load_bundle(
        Path(cfg.trained_model_bundle_path), device=device
    )

    afa_method = cast("AFAMethod", cast("object", afa_method))
    # TODO: need to check if this is the right function to call
    afa_method.set_cost_param(cfg.soft_budget_param)
    metadata: dict[str, Any] = manifest.get("metadata", {})
    metadata["resave_with_soft_budget"] = {
        "source_bundle": cfg.trained_model_bundle_path,
        "soft_budget_param": cfg.soft_budget_param,
    }

    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata=metadata,
    )
    log.info(f"Resaved bundle written to: {cfg.save_path}")


if __name__ == "__main__":
    main()
