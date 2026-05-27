"""Common configuration for all plotting scripts."""

from __future__ import annotations

# Plot dimensions
PLOT_WIDTH = 13
PLOT_HEIGHT = 5
PLOT_FONT_FAMILY = "Libertinus Serif"

# Method display names mapping
METHOD_NAME_MAPPING = {
    # Greedy
    "ma2018_external": "EDDI",
    "ma2018_builtin": "EDDI(builtin)",
    "covert2023": "GDFS",
    "gadgil2023": "DIME",
    # RL
    "jafa": "JAFA",
    "ol_with_mask": "OL(with-mask)",
    "ol_without_mask": "OL(without-mask)",
    "odin_model_based": "ODIN(model-based)",
    "odin_model_free": "ODIN(model-free)",
    # Oracle-based
    "aaco": "AACO",
    "aaco_nn": "AACO(NN)",
    # Static
    "permutation": "PT",
    "cae": "CAE",
    # Dummy
    "random_dummy": "Random",
    "sequential_dummy": "Sequential dummy",
}

METHOD_POLICY_FAMILY_MAPPING = {
    # EDDI / Ma 2018
    "ma2018_external": "ma2018",
    "ma2018_builtin": "ma2018",
    # Discriminative myopic
    "covert2023": "covert2023",
    "gadgil2023": "gadgil2023",
    # RL
    "jafa": "jafa",
    "ol_with_mask": "ol",
    "ol_without_mask": "ol",
    "odin_model_based": "odin",
    "odin_model_free": "odin",
    # Oracle-based
    "aaco": "aaco",
    "aaco_nn": "aaco",
    # Static
    "permutation": "static",
    "cae": "static",
    # Dummy / baselines
    "random_dummy": "dummy",
    "sequential_dummy": "dummy",
}

METHOD_FAMILY_COLOR_SCHEMES = {
    "tol_muted": {
        "ma2018": "#332288",  # indigo
        "covert2023": "#88CCEE",  # cyan
        "gadgil2023": "#44AA99",  # teal
        "jafa": "#117733",  # green
        "ol": "#999933",  # olive
        "odin": "#DDCC77",  # sand
        "aaco": "#CC6677",  # rose
        "static": "#882255",  # wine
        "dummy": "#AA4499",  # purple
    },
    "colorbrewer_set1": {
        "ma2018": "#E41A1C",  # red
        "covert2023": "#377EB8",  # blue
        "gadgil2023": "#4DAF4A",  # green
        "jafa": "#984EA3",  # purple
        "ol": "#FF7F00",  # orange
        "odin": "#FFFF33",  # yellow
        "aaco": "#A65628",  # brown
        "static": "#F781BF",  # pink
        "dummy": "#999999",  # grey
    },
    "colorbrewer_set2": {
        "ma2018": "#66C2A5",  # teal
        "covert2023": "#FC8D62",  # orange
        "gadgil2023": "#8DA0CB",  # blue
        "jafa": "#E78AC3",  # pink
        "ol": "#A6D854",  # green
        "odin": "#FFD92F",  # yellow
        "aaco": "#E5C494",  # tan
        "static": "#B3B3B3",  # grey
        "dummy": "#B3B3B3",  # grey
    },
    "colorbrewer_dark2": {
        "ma2018": "#1B9E77",  # teal
        "covert2023": "#D95F02",  # orange
        "gadgil2023": "#7570B3",  # purple
        "jafa": "#E7298A",  # pink
        "ol": "#66A61E",  # green
        "odin": "#E6AB02",  # mustard
        "aaco": "#A6761D",  # brown
        "static": "#666666",  # grey
        "dummy": "#666666",  # grey
    },
}


def get_method_color_mapping(scheme_name: str) -> dict[str, str]:
    """Return fixed method colors for a named qualitative color scheme."""
    family_colors = METHOD_FAMILY_COLOR_SCHEMES[scheme_name]
    return {
        method: family_colors[family]
        for method, family in METHOD_POLICY_FAMILY_MAPPING.items()
    }


METHOD_COLOR_SCHEMES = {
    name: get_method_color_mapping(name)
    for name in METHOD_FAMILY_COLOR_SCHEMES
}

# Change this value to switch the fixed policy colors used by plots.
ACTIVE_METHOD_COLOR_SCHEME = "colorbrewer_dark2"
METHOD_COLOR_MAPPING = METHOD_COLOR_SCHEMES[ACTIVE_METHOD_COLOR_SCHEME]

# Dataset display names mapping
DATASET_NAME_MAPPING = {
    "cube": "CUBE",
    "cube_nonuniform_costs": "CUBE-NUC",
    "cube_nm": "CUBE-NM",
    "cube_nm_without_noise": "CUBE-NM-noiseless",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "CUBE-noiseless",
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
    "cube_nm_pair": {
        "cube_nm",
        "cube_nm_without_noise",
    },
    "set1": {
        "cube",
        "cube_without_noise",
        "cube_nm",
        "cube_nm_without_noise",
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
        "fico",
        "pharyngitis",
    },
    "all": {
        "cube",
        "cube_without_noise",
        "cube_nm",
        "cube_nm_without_noise",
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
        "pharyngitis",
    },
    "kdd26": {
        "cube",
        "cube_nm",
        "cube_nm_without_noise",
        "bank_marketing",
        "miniboone",
        "actg",
        "physionet",
        "fashion_mnist",
        "imagenette",
        "ckd",
        "mnist",
        "cube_nonuniform_costs",
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
