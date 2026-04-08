"""Helpers for deriving stable dataset keys from class names."""

from __future__ import annotations

import re

_DATASET_KEY_ALIASES = {
    "ACTG175Dataset": "actg",
    "BankMarketingDataset": "bank_marketing",
    "CKDDataset": "ckd",
    "CubeNonUniformCostsDataset": "cube_nonuniform_costs",
    "FICODataset": "fico",
    "FashionMNISTDataset": "fashion_mnist",
    "ImagenetteDataset": "imagenette",
    "MiniBooNEDataset": "miniboone",
    "MNISTDataset": "mnist",
    "PharyngitisDataset": "pharyngitis",
    "PhysionetDataset": "physionet",
    "SyntheticMNISTDataset": "synthetic_mnist",
}


def camel_to_snake(name: str) -> str:
    """Convert CamelCase names to snake_case."""
    first_pass = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return second_pass.lower()


def infer_dataset_key_from_class_name(class_name: str) -> str:
    """Infer the canonical dataset key from a dataset class name."""
    if class_name in _DATASET_KEY_ALIASES:
        return _DATASET_KEY_ALIASES[class_name]
    return camel_to_snake(class_name.removesuffix("Dataset"))
