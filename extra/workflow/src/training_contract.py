"""
Render the training contract as command-line arguments.

The pipeline passes every pretraining and training script the same plain
`key=value` arguments (`docs/adr/0001-training-contract-as-library.md`).
The field names come from the contract dataclasses in
`afabench.training.contract`, which stay cheap to import (no torch) so the
Snakefile can import them at parse time.
"""

from collections.abc import Mapping
from dataclasses import fields

from afabench.training.contract import PretrainingContract, TrainingContract


def render_pretraining_contract(values: Mapping[str, object]) -> str:
    """Render the contract a pretraining script receives."""
    return _render(PretrainingContract, values)


def render_training_contract(values: Mapping[str, object]) -> str:
    """Render the contract a training script receives."""
    return _render(TrainingContract, values)


def _render(
    contract: type[PretrainingContract], values: Mapping[str, object]
) -> str:
    contract_fields = fields(contract)
    unknown = sorted(set(values) - {field.name for field in contract_fields})
    if unknown:
        message = f"Not fields of this training contract: {unknown}"
        raise ValueError(message)

    arguments: list[str] = []
    for field in contract_fields:
        value = values.get(field.name)
        if value is None:
            # A field that defaults to null, such as the pretrained model of
            # a method without a pretraining stage, is left off the command
            # line and keeps its default.
            if field.default is None:
                continue
            message = f"Training contract field {field.name!r} has no value."
            raise ValueError(message)
        arguments.append(f"{field.name}={value}")
    return " ".join(arguments)
