"""Lifecycle helpers around a pretraining or training run."""

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from afabench.core.bundle_system.bundle import Saveable, save_bundle
from afabench.core.utils import initialize_wandb_run, set_seed
from afabench.training.contract import (
    PretrainingContract,
    TrainingContract,
    TrainingStage,
)
from afabench.training.metric_logger import (
    MetricLogger,
    NullMetricLogger,
    WandbMetricLogger,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)


@contextmanager
def training_run(
    contract: PretrainingContract,
    stage: TrainingStage,
    *,
    tags: list[str],
    config: "DataclassInstance",
) -> Generator[MetricLogger]:
    """
    Seed, open a metric logger for `stage` and clean up afterwards.

    `config` is the full method config, usually the contract itself.
    """
    set_seed(contract.seed)
    if contract.use_wandb:
        metric_logger = WandbMetricLogger(
            initialize_wandb_run(
                cfg={**asdict(contract), **asdict(config)},
                job_type=stage,
                tags=tags,
            )
        )
    else:
        metric_logger = NullMetricLogger()

    try:
        yield metric_logger
    finally:
        metric_logger.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def save_result(
    obj: Saveable,
    contract: PretrainingContract,
    config: "DataclassInstance",
    *,
    stage: TrainingStage,
) -> None:
    """
    Save `obj` as a bundle at the contract's `save_path`.

    The bundle metadata records the stage, the contract fields and the
    remaining fields of `config`.
    """
    contract_class = (
        TrainingContract
        if isinstance(contract, TrainingContract)
        else PretrainingContract
    )
    contract_field_names = {f.name for f in fields(contract_class)}
    contract_dict = {
        name: value
        for name, value in asdict(contract).items()
        if name in contract_field_names
    }
    method_config_dict = {
        name: value
        for name, value in asdict(config).items()
        if name not in contract_field_names
    }
    save_bundle(
        obj=obj,
        path=Path(contract.save_path),
        metadata={
            "stage": stage,
            "contract": contract_dict,
            "method_config": method_config_dict,
        },
    )
    log.info(f"Saved {type(obj).__name__} to {contract.save_path}")
