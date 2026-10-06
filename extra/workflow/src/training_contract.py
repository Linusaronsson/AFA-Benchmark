"""
Render the training contract as command-line arguments.

The pipeline passes every pretraining and training script the same plain
`key=value` arguments (`docs/adr/0001-training-contract-as-library.md`).
This module is the second copy of the contract's field names; the first is
`afabench.training.contract`, which Snakemake cannot import.
`test/workflow/test_training_contract_arguments.py` keeps the copies equal.
"""

from collections.abc import Mapping, Sequence

PRETRAINING_CONTRACT_FIELDS: tuple[str, ...] = (
    "train_dataset_bundle_path",
    "val_dataset_bundle_path",
    "classifier_bundle_path",
    "save_path",
    "initializer",
    "unmasker",
    "dataset_key",
    "device",
    "seed",
    "use_wandb",
    "smoke_test",
)

TRAINING_CONTRACT_FIELDS: tuple[str, ...] = (
    *PRETRAINING_CONTRACT_FIELDS,
    "pretrained_model_bundle_path",
    "hard_budget",
    "soft_budget_param",
)

# Left out of the command line for methods without a pretraining stage; the
# contract dataclass then keeps its default of null.
OPTIONAL_TRAINING_CONTRACT_FIELDS: frozenset[str] = frozenset(
    {"pretrained_model_bundle_path"}
)


def render_pretraining_contract(values: Mapping[str, object]) -> str:
    """Render the contract a pretraining script receives."""
    return _render(PRETRAINING_CONTRACT_FIELDS, values, optional=frozenset())


def render_training_contract(values: Mapping[str, object]) -> str:
    """Render the contract a training script receives."""
    return _render(
        TRAINING_CONTRACT_FIELDS,
        values,
        optional=OPTIONAL_TRAINING_CONTRACT_FIELDS,
    )


def _render(
    field_names: Sequence[str],
    values: Mapping[str, object],
    *,
    optional: frozenset[str],
) -> str:
    unknown = sorted(set(values) - set(field_names))
    if unknown:
        message = f"Not fields of this training contract: {unknown}"
        raise ValueError(message)

    arguments: list[str] = []
    for name in field_names:
        value = values.get(name)
        if value is None:
            if name in optional:
                continue
            message = f"Training contract field {name!r} has no value."
            raise ValueError(message)
        arguments.append(f"{name}={value}")
    return " ".join(arguments)
