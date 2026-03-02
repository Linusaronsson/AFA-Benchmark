"""Common configuration for all plotting scripts."""

from __future__ import annotations

# Plot dimensions
PLOT_WIDTH = 13
PLOT_HEIGHT = 5

# Method display names mapping
METHOD_NAME_MAPPING = {
    # Greedy
    "ma2018_external": "EDDI-GM",
    "ma2018_builtin": "EDDI-GM+builtin",
    "covert2023": "GDFS-DM",
    "gadgil2023": "DIME-DM",
    # RL
    "jafa": "JAFA-MFRL",
    "ol_with_mask": "OL-MFRL",
    "ol_without_mask": "OL-MFRL+no-mask",
    "odin_model_based": "ODIN-MBRL",
    "odin_model_free": "ODIN-MFRL",
    # Oracle-based
    "aaco": "AACO",
    "aaco_nn": "AACO+NN",
    # Static
    "permutation": "PT-S",
    "cae": "CAE-S",
    # Dummy
    "random_dummy": "Random",
    "sequential_dummy": "Sequential dummy",
    "stop_baseline": "No acquisition",
    "stop_baseline_marf": "No acquisition (MARF)",
    "stop_baseline_madt": "No acquisition (MADT)",
    "stop_baseline_magbt": "No acquisition (MAGBT)",
    "stop_baseline_malasso": "No acquisition (MALasso)",
    "aaco_marf": "AACO (MARF)",
    "aaco_madt": "AACO (MADT)",
    "aaco_magbt": "AACO (MAGBT)",
    "aaco_malasso": "AACO (MALasso)",
    "aaco_full": "AACO (full train)",
    "aaco_zero_fill": "AACO (zero-fill train)",
    "aaco_impute_mean": "AACO (mean-impute train)",
    "aaco_mask_aware": "AACO (mask-aware train)",
}

# Dataset display names mapping
DATASET_NAME_MAPPING = {
    "cube": "CUBE",
    "cube_nonuniform_costs": "CUBE-NUC",
    "afa_context": "AFAContext",
    "afa_context_v2": "CUBE-NM",
    "afa_context_v2_without_noise": "CUBE-NM-noiseless",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "CUBE-noiseless",
    "afa_context_without_noise": "AFAContext-noiseless",
    "synthetic_mnist_without_noise": "Synthetic-MNIST-noiseless",
    "mnist": "MNIST",
    "actg": "ACTG",
    "bank_marketing": "BankMarketing",
    "ckd": "CKD",
    "diabetes": "Diabetes",
    "fico": "FICO",
    "fashion_mnist": "FashionMNIST",
    "miniboone": "MiniBooNE",
    "pharyngitis": "Pharyngitis",
    "physionet": "PhysioNet",
    "imagenette": "Imagenette",
}

# Datasets that use F1 score instead of accuracy
DATASETS_WITH_F_SCORE = ["physionet", "bank_marketing"]

# Dataset groupings for organized plotting
DATASET_SETS = {
    "context_v2_pair": {
        "afa_context_v2",
        "afa_context_v2_without_noise",
    },
    "set1": {
        "cube",
        "cube_without_noise",
        "afa_context",
        "afa_context_without_noise",
        "afa_context_v2_without_noise",
        "miniboone",
        "bank_marketing",
        "diabetes",
        "physionet",
        "actg",
        "fashion_mnist",
    },
    "set2": {
        "imagenette",
        "ckd",
        "mnist",
        "cube_nonuniform_costs",
        "afa_context_v2",
        "fico",
        "pharyngitis",
    },
    "all": {
        "cube",
        "cube_without_noise",
        "afa_context",
        "afa_context_without_noise",
        "afa_context_v2_without_noise",
        "miniboone",
        "bank_marketing",
        "diabetes",
        "physionet",
        "actg",
        "fashion_mnist",
        "fico",
        "imagenette",
        "ckd",
        "mnist",
        "cube_nonuniform_costs",
        "afa_context_v2",
        "pharyngitis",
    },
}

# Default color palette for discrete visualization
# Using RColorBrewer 'Dark2' palette (color-blind friendly, 8 colors)
# Available options:
#   - 'Dark2' (default): Dark, color-blind friendly, 8 colors
#   - 'Set2': Medium saturation, color-blind friendly, 8 colors
#   - 'Set1': Bold, saturated, 9 colors
#   - 'Accent': Accent colors, 8 colors
#   - 'Paired': Paired colors, 12 colors
#   - 'Set3': Pastel, 12 colors
COLOR_PALETTE_NAME = "Dark2"
