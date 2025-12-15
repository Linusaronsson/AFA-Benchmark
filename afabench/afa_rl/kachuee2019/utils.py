from jaxtyping import Float
from torch import Tensor

from afabench.afa_rl.kachuee2019.models import (
    Kachuee2019PQModule,
    LitKachuee2019PQModule,
)
from afabench.common.config_classes import Kachuee2019PretrainConfig


def get_kachuee2019_model_from_config(
    cfg: Kachuee2019PretrainConfig,
    feature_shape: torch.Size,
    n_classes: int,
    class_probabilities: Float[Tensor, "n_classes"],
) -> LitKachuee2019PQModule:
    pq_module = Kachuee2019PQModule(
        n_features=feature_shape.numel(), n_classes=n_classes, cfg=cfg.pq_module
    )
    lit_model = LitKachuee2019PQModule(
        pq_module=pq_module,
        class_probabilities=class_probabilities,
        n_feature_dims=len(feature_shape),
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )
    return lit_model
