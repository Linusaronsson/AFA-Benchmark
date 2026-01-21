"""
Configuration loading and validation for Snakemake workflows.

Provides utilities for loading and processing Snakemake configuration files
with support for methods, datasets, budgets, and other common parameters.
"""

NO_PRETRAIN_STR = "NO_PRETRAIN"


def load_config(config):
    """
    Load and validate configuration variables from the Snakemake config.

    Processes and validates:
    - Basic configuration (device, seeds, etc.)
    - Method options and filtering
    - Dataset configuration
    - Budget and unmasker parameters
    - Budget parameter combinations

    Args:
        config: Snakemake config dictionary

    Returns:
        Dictionary containing all processed configuration variables.
    """
    # ========================================================================
    # Basic Configuration
    # ========================================================================

    dataset_instance_indices = config.get("dataset_instance_indices", (0, 1))
    initializer = config.get("initializer", "cold")
    eval_dataset_split = config.get("eval_dataset_split", "val")
    device = config.get("device", "cpu")
    use_wandb = config.get("use_wandb", False)
    smoke_test = config.get("smoke_test", False)

    # ========================================================================
    # Methods Configuration
    # ========================================================================

    method_options = config.get("method_options", None)
    if method_options is None:
        raise ValueError("Expected method_options to be provided.")

    methods = config.get("methods", [])
    if methods is None:
        raise ValueError("Expected methods to be provided.")

    # Filter methods by pretraining stage availability
    # A method has a pretraining stage if pretrain_script_name is set
    methods_with_pretraining_stage = [
        method
        for method, options in method_options.items()
        if "pretrain_script_name" in options and method in methods
    ]

    methods_without_pretraining_stage = [
        method
        for method, options in method_options.items()
        if "pretrain_script_name" not in options and method in methods
    ]

    # Build method option mappings for training scripts
    method_train_script_names = {
        method: options.get("train_script_name", method)
        for method, options in method_options.items()
        if method in methods and "train_script_name" in options
    }

    # Build method option mappings for pretraining scripts
    method_pretrain_script_names = {
        method: options.get("pretrain_script_name", method)
        for method, options in method_options.items()
        if method in methods and "pretrain_script_name" in options
    }

    # Default method_specific_params to empty list if not provided
    method_specific_params = {
        method: " ".join(options.get("method_specific_params", []))
        for method, options in method_options.items()
        if method in methods
    }

    # ========================================================================
    # Dataset Configuration
    # ========================================================================

    datasets = config.get("datasets", None)
    if datasets is None:
        raise ValueError("Expected datasets to be provided.")

    # ========================================================================
    # Budget and Unmasker Configuration
    # ========================================================================

    unmaskers_raw = config.get("unmaskers", None)
    if unmaskers_raw is None:
        raise ValueError("Expected unmaskers to be provided.")
    unmaskers = _fill_missing_datasets_with_default(unmaskers_raw, datasets)

    hard_budgets_raw = config.get("hard_budgets", None)
    if hard_budgets_raw is None:
        raise ValueError("Expected hard_budgets to be provided.")
    hard_budgets = _fill_missing_datasets_with_default(
        hard_budgets_raw, datasets
    )

    soft_budget_params_raw = config.get("soft_budget_params", None)
    if soft_budget_params_raw is None:
        raise ValueError("Expected soft_budget_params to be provided.")

    # Fill in missing datasets for each method's soft budget params
    soft_budget_params = {
        method: _fill_missing_datasets_with_default(method_params, datasets)
        for method, method_params in soft_budget_params_raw.items()
    }

    # ========================================================================
    # Combined Budget Parameters
    # ========================================================================

    # Create combinations of hard and soft budget parameters for each method-dataset pair.
    # Methods can be trained with either a hard budget or a soft budget, but not both.
    # Note: "null" is used instead of None since it will be passed to hydra
    hard_budget_and_soft_budget_params = {
        method: {
            dataset: _create_budget_combinations(
                method,
                dataset,
                hard_budgets[dataset],
                soft_budget_params[method][dataset],
            )
            for dataset in datasets
        }
        for method in methods
    }

    return {
        "NO_PRETRAIN_STR": NO_PRETRAIN_STR,
        "DATASET_INSTANCE_INDICES": dataset_instance_indices,
        "INITIALIZER": initializer,
        "EVAL_DATASET_SPLIT": eval_dataset_split,
        "DEVICE": device,
        "USE_WANDB": use_wandb,
        "SMOKE_TEST": smoke_test,
        "METHOD_OPTIONS": method_options,
        "METHODS": methods,
        "METHODS_WITH_PRETRAINING_STAGE": methods_with_pretraining_stage,
        "METHODS_WITHOUT_PRETRAINING_STAGE": methods_without_pretraining_stage,
        "METHOD_TRAIN_SCRIPT_NAMES": method_train_script_names,
        "METHOD_PRETRAIN_SCRIPT_NAMES": method_pretrain_script_names,
        "METHOD_SPECIFIC_PARAMS": method_specific_params,
        "DATASETS": datasets,
        "UNMASKERS": unmaskers,
        "HARD_BUDGETS": hard_budgets,
        "SOFT_BUDGET_PARAMS": soft_budget_params,
        "HARD_BUDGET_AND_SOFT_BUDGET_PARAMS": hard_budget_and_soft_budget_params,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _fill_missing_datasets_with_default(config_dict, datasets):
    """Fill in missing datasets with the default value."""
    return config_dict | {
        dataset: config_dict["default"]
        for dataset in datasets
        if dataset not in config_dict
    }


def _create_budget_combinations(
    method, dataset, hard_budgets, soft_budget_params
):
    """Create combinations of hard and soft budget parameters for a method-dataset pair."""
    hard_budget_combos = [(hb, "null") for hb in hard_budgets]
    soft_budget_combos = [("null", sbp) for sbp in soft_budget_params]
    return hard_budget_combos + soft_budget_combos
