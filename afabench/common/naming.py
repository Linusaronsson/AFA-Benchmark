"""Helpers for deriving stable dataset keys from class names."""

from __future__ import annotations

import re

_DATASET_KEY_ALIASES = {
    "ACTG175Dataset": "actg",
    "BankMarketingDataset": "bank_marketing",
    "CKDDataset": "ckd",
    "CubeNMARDataset": "cube_nm_ar",
    "CubeNonUniformCostsDataset": "cube_nonuniform_costs",
    "FICODataset": "fico",
    "FashionMNISTDataset": "fashion_mnist",
    "ImagenetteDataset": "imagenette",
    "MiniBooNEDataset": "miniboone",
    "MNISTDataset": "mnist",
    "PharyngitisDataset": "pharyngitis",
    "PhysionetDataset": "physionet",
    "SyntheticMNISTDataset": "synthetic_mnist",
    "XORNoisyShortcutDataset": "xor_noisy_shortcut",
}

LEGACY_DATASET_KEY_ALIASES = {
    "afa_context": "cube_nm_3ctx",
    "afa_context_without_noise": "cube_nm_3ctx_without_noise",
    "afa_context_v2": "cube_nm",
    "afa_context_v2_without_noise": "cube_nm_without_noise",
}


def camel_to_snake(name: str) -> str:
    """Convert CamelCase names to snake_case."""
    first_pass = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return second_pass.lower()


def infer_dataset_key_from_class_name(class_name: str) -> str:
    """Infer the canonical dataset key from a dataset class name."""
    if class_name in _DATASET_KEY_ALIASES:
        return canonicalize_dataset_key(_DATASET_KEY_ALIASES[class_name])
    return canonicalize_dataset_key(
        camel_to_snake(class_name.removesuffix("Dataset"))
    )


def canonicalize_dataset_key(dataset_key: str) -> str:
    """Collapse historical dataset ids onto the current canonical keys."""
    return LEGACY_DATASET_KEY_ALIASES.get(dataset_key, dataset_key)
