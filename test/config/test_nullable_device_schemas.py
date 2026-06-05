from dataclasses import is_dataclass

import pytest
from omegaconf import OmegaConf

from afabench.components.classifiers.config import (
    TrainMaskedViTClassifierConfig,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMEPretrainingConfig,
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSPretrainingConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.generative.eddi.config import (
    EDDITrainingConfig,
)
from afabench.components.methods.static.cae.config import CAETrainingConfig
from afabench.components.methods.static.pt.config import (
    PermutationTrainingConfig,
)


@pytest.mark.parametrize(
    ("config_type", "nullable_fields"),
    [
        (TrainMaskedViTClassifierConfig, ["device", "seed"]),
        (DIMEPretrainingConfig, ["device", "seed"]),
        (DIMETrainingConfig, ["device", "seed", "hard_budget"]),
        (GDFSPretrainingConfig, ["device", "seed"]),
        (GDFSTrainingConfig, ["device", "seed", "hard_budget"]),
        (EDDITrainingConfig, ["device", "seed", "hard_budget"]),
        (CAETrainingConfig, ["device", "seed", "hard_budget"]),
        (PermutationTrainingConfig, ["device", "seed", "hard_budget"]),
    ],
)
def test_nullable_defaults_match_schema(
    config_type: type,
    nullable_fields: list[str],
) -> None:
    assert is_dataclass(config_type)
    cfg = OmegaConf.structured(config_type)

    merged = OmegaConf.merge(
        cfg,
        dict.fromkeys(nullable_fields),
    )

    for field_name in nullable_fields:
        assert getattr(merged, field_name) is None
