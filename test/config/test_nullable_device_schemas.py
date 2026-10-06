from dataclasses import is_dataclass

import pytest
from omegaconf import OmegaConf

from afabench.components.classifiers.config import (
    TrainMaskedViTClassifierConfig,
)
from afabench.components.methods.discriminative.dime.config import (
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.config import (
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
        (DIMETrainingConfig, ["hard_budget", "soft_budget_param"]),
        (GDFSTrainingConfig, ["hard_budget", "soft_budget_param"]),
        (EDDITrainingConfig, ["hard_budget", "soft_budget_param"]),
        (CAETrainingConfig, ["hard_budget", "soft_budget_param"]),
        (PermutationTrainingConfig, ["hard_budget", "soft_budget_param"]),
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
