"""Provenance records for bundles that tests write as fixtures."""

from collections.abc import Sequence

from afabench.core.provenance import (
    ProvenanceInput,
    ProvenanceRecord,
    Split,
    Stage,
    capture_provenance,
)


def placeholder_provenance(
    stage: Stage = "dataset_generation",
    *,
    seed: int = 0,
    dataset_key: str | None = "cube",
    dataset_realization_index: int | None = 0,
    split: Split | None = None,
    method_name: str | None = None,
    inputs: Sequence[ProvenanceInput] = (),
) -> ProvenanceRecord:
    """
    Capture a record for a fixture bundle, with a placeholder config.

    Built through `capture_provenance` like production records (ADR 0002),
    so the code, environment and compute facts are real.
    """
    return capture_provenance(
        stage=stage,
        resolved_config={"placeholder": True},
        seed=seed,
        smoke_test=True,
        device="cpu",
        inputs=inputs,
        method_name=method_name,
        dataset_key=dataset_key,
        dataset_realization_index=dataset_realization_index,
        split=split,
    )
