"""Helpers for deriving stable dataset keys from class names."""

from __future__ import annotations

import re

from afabench.datasets.aliases import DATASET_KEY_ALIASES


def camel_to_snake(name: str) -> str:
    """Convert CamelCase names to snake_case."""
    first_pass = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return second_pass.lower()


def infer_dataset_key_from_class_name(class_name: str) -> str:
    """Infer the canonical dataset key from a dataset class name."""
    if class_name in DATASET_KEY_ALIASES:
        return DATASET_KEY_ALIASES[class_name]
    return camel_to_snake(class_name.removesuffix("Dataset"))
