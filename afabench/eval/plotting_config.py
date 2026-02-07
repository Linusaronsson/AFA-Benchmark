"""Common configuration for all plotting scripts."""

from __future__ import annotations

# Plot dimensions
PLOT_WIDTH = 13
PLOT_HEIGHT = 5

# Method display names mapping
METHOD_NAME_MAPPING = {
    # Dummy
    "random_dummy": "Random",
    "sequential_dummy": "Sequential dummy",
    # RL
    "jafa": "JAFA",
    "odin_model_based": "ODIN-MB",
    "odin_model_free": "ODIN-MF",
    "ol_without_mask": "OL",
    "ol_with_mask": "OL+mask",
    "eddi": "EDDI",
    "dime": "DIME",
    "aaco": "AACO",
    "aaco_nn": "AACO+NN",
    # Static
    "cae": "CAE",
    "permutation": "Permutation",
    # Greedy
    "covert2023": "DIME",
    "gadgil2023": "GDFS",
    "ma2018_builtin": "EDDI-builtin",
    "ma2018_external": "EDDI-external",
}

# Dataset display names mapping
DATASET_NAME_MAPPING = {
    "cube": "CUBE",
    "cube_nonuniform_costs": "Cube (non-uniform costs)",
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
    "fashion_mnist": "FashionMNIST",
    "miniboone": "MiniBooNE",
    "physionet": "PhysioNet",
    "imagenette": "Imagenette",
}

# Datasets that use F1 score instead of accuracy
DATASETS_WITH_F_SCORE = ["physionet", "bank_marketing"]

# Dataset groupings for organized plotting
DATASET_SETS = {
    "set1": {
        "cube",
        "cube_nonuniform_costs",
        "afa_context_v2",
        "afa_context_v2_without_noise",
        "mnist",
        "imagenette",
        "miniboone",
        "physionet",
    },
    "set2": {
        "actg",
        "bank_marketing",
        "ckd",
        "diabetes",
        "fashion_mnist",
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
