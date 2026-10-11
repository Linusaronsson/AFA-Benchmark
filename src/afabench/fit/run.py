"""Lifecycle helpers around a fit-stage (pretraining or training) run."""

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from afabench.core.bundle_system.bundle import Saveable, save_bundle
from afabench.core.provenance import ProvenanceRecord, capture_provenance
from afabench.core.utils import initialize_wandb_run, set_seed
from afabench.fit.contract import (
    BaseContract,
    FitStage,
    PretrainingContract,
    TrainingContract,
)
from afabench.fit.inputs import FitInputs
from afabench.fit.metric_logger import (
    MetricLogger,
    NullMetricLogger,
    WandbMetricLogger,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)

_CONTRACT_CLASSES: dict[FitStage, type[BaseContract]] = {
    "pretraining": PretrainingContract,
    "training": TrainingContract,
}


@contextmanager
def fit_run(
    contract: BaseContract,
    *,
    tags: list[str],
    config: "DataclassInstance | None" = None,
    inputs: FitInputs | None = None,
) -> Generator[MetricLogger]:
    """
    Seed, open a metric logger for the contract's stage and clean up afterwards.

    `config` is the full method config; it defaults to the contract, which
    method configs inherit. CUDA
    memory is released only when the contract's device is a CUDA device, so
    a CPU run never initialises a CUDA context. Dataset bundles whose
    records disagree on the dataset identity raise
    `DatasetIdentityMismatchError` on entry, before any training. `inputs`
    are the contract's `FitInputs` the script loads its inputs from, so
    that their manifests are read once; without them, fresh ones are used.
    """
    (FitInputs(contract) if inputs is None else inputs).dataset_identity()
    set_seed(contract.seed)
    if contract.use_wandb:
        metric_logger = WandbMetricLogger(
            initialize_wandb_run(
                cfg={
                    **asdict(contract),
                    **asdict(contract if config is None else config),
                },
                job_type=contract.stage,
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
        if torch.device(contract.device).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def save_result(
    obj: Saveable,
    contract: BaseContract,
    config: "DataclassInstance | None" = None,
    *,
    inputs: FitInputs | None = None,
) -> None:
    """
    Save `obj` as a bundle at the contract's `save_path`.

    The bundle metadata records the stage, the contract fields and the
    remaining fields of `config`, which defaults to the contract as for
    `fit_run`. The bundle's provenance record
    (ADR 0002) takes the contract's seed, its input bundles and, for
    training, its method name; the dataset identity is copied from the
    dataset bundles' own records. `inputs` are the contract's `FitInputs`,
    as for `fit_run`.
    """
    contract_class = _CONTRACT_CLASSES[contract.stage]
    contract_field_names = {f.name for f in fields(contract_class)}
    contract_dict = {
        name: value
        for name, value in asdict(contract).items()
        if name in contract_field_names
    }
    method_config_dict = {
        name: value
        for name, value in asdict(
            contract if config is None else config
        ).items()
        if name not in contract_field_names
    }
    save_bundle(
        obj=obj,
        path=Path(contract.save_path),
        metadata={
            "stage": contract.stage,
            "contract": contract_dict,
            "method_config": method_config_dict,
        },
        provenance=_fit_provenance(
            contract,
            FitInputs(contract) if inputs is None else inputs,
            resolved_config={**contract_dict, **method_config_dict},
        ),
    )
    log.info(f"Saved {type(obj).__name__} to {contract.save_path}")


def _fit_provenance(
    contract: BaseContract,
    inputs: FitInputs,
    *,
    resolved_config: dict[str, object],
) -> ProvenanceRecord:
    return capture_provenance(
        stage=contract.stage,
        resolved_config=resolved_config,
        seed=contract.seed,
        smoke_test=contract.smoke_test,
        device=contract.device,
        inputs=inputs.provenance_inputs(),
        dataset_identity=inputs.dataset_identity(),
        method_name=(
            contract.method_name
            if isinstance(contract, TrainingContract)
            else None
        ),
    )
