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
    - Pretrained model configurations
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
    # Pretrained Models Configuration
    # ========================================================================

    pretrain_mapping = config.get("pretrain_mapping", None)
    if pretrain_mapping is None:
        raise ValueError("Expected pretrain_mapping to be provided.")

    # Extract pretrain script names and params for each pretrained model
    pretrain_model_script_names = {
        model_name: model_config["pretrain_script_name"]
        for model_name, model_config in pretrain_mapping.items()
    }

    pretrain_model_params = {
        model_name: " ".join(model_config.get("pretrain_params", []))
        for model_name, model_config in pretrain_mapping.items()
    }

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
    # A method has a pretraining stage if pretrained_model_name is set
    methods_with_pretraining_stage = [
        method
        for method, options in method_options.items()
        if "pretrained_model_name" in options and method in methods
    ]

    methods_without_pretraining_stage = [
        method
        for method, options in method_options.items()
        if "pretrained_model_name" not in options and method in methods
    ]

    # Build method option mappings for training scripts
    method_train_script_names = {
        method: options.get("train_script_name", method)
        for method, options in method_options.items()
        if method in methods and "train_script_name" in options
    }

    # Build mapping from method to pretrained model name
    method_to_pretrained_model = {
        method: options["pretrained_model_name"]
        for method, options in method_options.items()
        if method in methods and "pretrained_model_name" in options
    }

    # Filter pretrain_names to only include those needed by selected methods
    pretrain_names_needed = set(method_to_pretrained_model.values())
    pretrain_names = [
        name
        for name in pretrain_mapping.keys()
        if name in pretrain_names_needed
    ]

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

    eval_hard_budgets_raw = config.get("eval_hard_budgets", None)
    if eval_hard_budgets_raw is None:
        raise ValueError("Expected eval_hard_budgets to be provided.")
    eval_hard_budgets = _fill_missing_datasets_with_default(
        eval_hard_budgets_raw, datasets
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
    # Train Budget Mapping Configuration
    # ========================================================================

    # Extract eval_to_train_hard_budget_mapping from method_options
    # Format: {method: {dataset: {eval_budget: train_budget, ...}}}
    eval_to_train_hard_budget_mapping = {
        method: options.get("eval_to_train_hard_budget_mapping", {})
        for method, options in method_options.items()
        if method in methods
    }

    # ========================================================================
    # Budget Parameters Structure
    # ========================================================================

    # Build the main data structure: method -> dataset -> list of
    # (train_hard_budget, eval_hard_budget, soft_budget_param) tuples
    #
    # For each method-dataset pair:
    # - Create hard budget combinations: (train_budget, eval_budget, "null")
    # - Create soft budget combinations: ("null", "null", soft_budget_param)
    budget_params = {
        method: {
            dataset: _create_budget_combinations(
                method,
                dataset,
                eval_hard_budgets[dataset],
                soft_budget_params[method][dataset],
                eval_to_train_hard_budget_mapping,
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
        "PRETRAIN_NAMES": pretrain_names,
        "PRETRAIN_SCRIPT_NAMES": pretrain_model_script_names,
        "PRETRAIN_PARAMS": pretrain_model_params,
        "METHOD_OPTIONS": method_options,
        "METHODS": methods,
        "METHODS_WITH_PRETRAINING_STAGE": methods_with_pretraining_stage,
        "METHODS_WITHOUT_PRETRAINING_STAGE": methods_without_pretraining_stage,
        "METHOD_TRAIN_SCRIPT_NAMES": method_train_script_names,
        "METHOD_TO_PRETRAINED_MODEL": method_to_pretrained_model,
        "METHOD_SPECIFIC_PARAMS": method_specific_params,
        "DATASETS": datasets,
        "UNMASKERS": unmaskers,
        "BUDGET_PARAMS": budget_params,
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


def _get_eval_hard_budget(
    method,
    dataset,
    train_hard_budget,
    eval_to_train_hard_budget_mapping,
):
    """
    Get the evaluation hard budget for a given training hard budget.

    If the method has a mapping defined, use it. Otherwise, use the training budget.

    Args:
        method: Method name
        dataset: Dataset name
        train_hard_budget: Training hard budget value
        eval_to_train_hard_budget_mapping: Mapping from eval to train budgets

    Returns:
        The evaluation hard budget to use
    """
    # Check if method has a mapping for this dataset
    method_mapping = eval_to_train_hard_budget_mapping.get(method, {})
    dataset_mapping = method_mapping.get(dataset, {})

    # Search for train_hard_budget in the mapping values (reverse lookup)
    for eval_budget, train_budget in dataset_mapping.items():
        if train_budget == train_hard_budget:
            return eval_budget

    # Default: use the training budget for evaluation
    return train_hard_budget


def _create_budget_combinations(
    method,
    dataset,
    eval_hard_budgets,
    soft_budget_params,
    eval_to_train_hard_budget_mapping,
):
    """
    Create budget parameter tuples for a method-dataset pair.

    Returns list of tuples: (train_hard_budget, eval_hard_budget, soft_budget_param)

    For hard budgets: train_hard_budget and eval_hard_budget are mapped per config,
    soft_budget_param is "null"

    For soft budgets: train_hard_budget and eval_hard_budget are "null",
    soft_budget_param is the actual soft budget value
    """
    result = []

    # Hard budget combinations
    for eval_budget in eval_hard_budgets:
        train_budget = _get_train_hard_budget_from_eval(
            method, dataset, eval_budget, eval_to_train_hard_budget_mapping
        )
        result.append((train_budget, eval_budget, "null"))

    # Soft budget combinations
    for soft_budget in soft_budget_params:
        result.append(("null", "null", soft_budget))

    return result


def _get_train_hard_budget_from_eval(
    method, dataset, eval_budget, eval_to_train_hard_budget_mapping
):
    """
    Get the training hard budget for a given evaluation hard budget.

    This is the reverse lookup: given an eval budget, find the train budget.

    Args:
        method: Method name
        dataset: Dataset name
        eval_budget: Evaluation hard budget value
        eval_to_train_hard_budget_mapping: Mapping from eval to train budgets

    Returns:
        The training hard budget to use (or same as eval_budget if no mapping)
    """
    # Check if method has a mapping for this dataset
    method_mapping = eval_to_train_hard_budget_mapping.get(method, {})
    dataset_mapping = method_mapping.get(dataset, {})

    # Direct lookup: eval_budget -> train_budget
    if eval_budget in dataset_mapping:
        return dataset_mapping[eval_budget]

    # Default: use the eval budget for training (same budget)
    return eval_budget
