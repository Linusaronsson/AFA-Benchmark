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
    eval_dataset_split = config.get(
        "eval_dataset_split", "test"
    )  # switch to val while developing, and train if debugging
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

    method_sets = config.get("method_sets", {})
    # Filter out methods that have not been enabled by the "methods" option
    # and remove empty method_sets
    filtered_method_sets = {}
    for key in method_sets:
        filtered_methods = [
            method for method in method_sets[key] if method in methods
        ]
        if filtered_methods:
            filtered_method_sets[key] = filtered_methods
    method_sets = filtered_method_sets

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

    is_method_using_max_hard_budget_when_training_soft_budget = {
        method: options.get(
            "use_max_hard_budget_when_training_soft_budget", False
        )
        for method, options in method_options.items()
        if method in methods
    }

    # Extract hard_budget_ignored_datasets from method_options
    # Format: {method: [dataset1, dataset2, ...]}
    hard_budget_ignored_datasets = {
        method: options.get("hard_budget_ignored_datasets", [])
        for method, options in method_options.items()
        if method in methods
    }

    # Extract soft_budget_ignored_datasets from method_options
    # Format: {method: [dataset1, dataset2, ...]}
    soft_budget_ignored_datasets = {
        method: options.get("soft_budget_ignored_datasets", [])
        for method, options in method_options.items()
        if method in methods
    }

    # ========================================================================
    # Dataset Configuration
    # ========================================================================

    datasets = config.get("datasets", None)
    if datasets is None:
        raise ValueError("Expected datasets to be provided.")

    # Extract eval_batch_size from method_options
    # Format: {method: {dataset: batch_size}}
    eval_batch_sizes = {}
    for method, options in method_options.items():
        if method in methods:
            batch_size_config = options.get("eval_batch_size")
            if batch_size_config is None:
                # If eval_batch_size is not specified, use batch size of 1 for all
                eval_batch_sizes[method] = dict.fromkeys(datasets, 1)
            elif isinstance(batch_size_config, dict):
                # Fill in missing datasets with the default batch size
                # If no default is specified, use batch size of 1
                default_batch_size = batch_size_config.get("default", 1)
                eval_batch_sizes[method] = batch_size_config | {
                    dataset: default_batch_size
                    for dataset in datasets
                    if dataset not in batch_size_config
                }
            else:
                # If eval_batch_size is a scalar, use it for all datasets
                eval_batch_sizes[method] = dict.fromkeys(datasets, batch_size_config)

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
    # Classifier Names Configuration
    # ========================================================================

    classifier_names_raw = config.get("classifier_names", None)
    if classifier_names_raw is None:
        raise ValueError("Expected classifier_names to be provided.")
    classifier_names = _fill_missing_datasets_with_default(
        classifier_names_raw, datasets
    )

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
    # (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) tuples
    #
    # For each method-dataset pair:
    # - Create hard budget combinations: (train_budget, eval_budget, "null", "null")
    # - Create soft budget combinations: ("null", "null", train_soft_budget_param, eval_soft_budget_param)
    budget_params = {
        method: {
            dataset: _create_budget_combinations(
                method,
                dataset,
                eval_hard_budgets[dataset],
                soft_budget_params[method][dataset],
                eval_to_train_hard_budget_mapping,
                use_max_hard_budget_when_training_soft_budget=is_method_using_max_hard_budget_when_training_soft_budget[
                    method
                ],
                ignore_hard_budgets=dataset
                in hard_budget_ignored_datasets[method],
                ignore_soft_budgets=dataset
                in soft_budget_ignored_datasets[method],
            )
            for dataset in datasets
        }
        for method in methods
    }

    # Compute which datasets are actually used for each method
    # (not fully ignored by both hard and soft budgets)
    datasets_used_per_method = _compute_datasets_used_per_method(
        methods,
        datasets,
        hard_budget_ignored_datasets,
        soft_budget_ignored_datasets,
    )

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
        "CLASSIFIER_NAMES": classifier_names,
        "METHOD_SETS": method_sets,
        "EVAL_BATCH_SIZES": eval_batch_sizes,
        "HARD_BUDGET_IGNORED_DATASETS": hard_budget_ignored_datasets,
        "SOFT_BUDGET_IGNORED_DATASETS": soft_budget_ignored_datasets,
        "DATASETS_USED_PER_METHOD": datasets_used_per_method,
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
    method,
    dataset,
    eval_hard_budgets,
    soft_budget_params,
    eval_to_train_hard_budget_mapping,
    use_max_hard_budget_when_training_soft_budget: bool = False,
    ignore_hard_budgets: bool = False,
    ignore_soft_budgets: bool = False,
):
    """
    Create budget parameter tuples for a method-dataset pair.

    Returns list of tuples: (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param)

    For hard budgets: train_hard_budget and eval_hard_budget are mapped per config,
    soft budget params are "null"

    For soft budgets: hard budgets are "null",
    soft budget params are extracted from the tuple: [train_soft_budget_param, eval_soft_budget_param]

    If ignore_hard_budgets is True, hard budget combinations are skipped.
    If ignore_soft_budgets is True, soft budget combinations are skipped.
    """
    result = []
    mapped_train_budgets = [
        _get_train_hard_budget_from_eval(
            method, dataset, eval_budget, eval_to_train_hard_budget_mapping
        )
        for eval_budget in eval_hard_budgets
    ]
    max_train_hard_budget = (
        max(mapped_train_budgets) if mapped_train_budgets else "null"
    )

    # Hard budget combinations (skip if ignore_hard_budgets is True)
    if not ignore_hard_budgets:
        for eval_budget in eval_hard_budgets:
            train_budget = _get_train_hard_budget_from_eval(
                method, dataset, eval_budget, eval_to_train_hard_budget_mapping
            )
            result.append((train_budget, eval_budget, "null", "null"))

    # Soft budget combinations (skip if ignore_soft_budgets is True)
    if not ignore_soft_budgets:
        for soft_budget_tuple in soft_budget_params:
            # soft_budget_tuple is [train_soft_budget_param, eval_soft_budget_param]
            train_soft_budget_param = _normalize_nullable_param(
                soft_budget_tuple[0]
            )
            eval_soft_budget_param = _normalize_nullable_param(
                soft_budget_tuple[1]
            )
            train_hard_budget = (
                max_train_hard_budget
                if use_max_hard_budget_when_training_soft_budget
                else "null"
            )
            result.append(
                (
                    train_hard_budget,
                    "null",
                    train_soft_budget_param,
                    eval_soft_budget_param,
                )
            )

    return result


def _normalize_nullable_param(value):
    if value is None:
        return "null"
    return value


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


def _compute_datasets_used_per_method(
    methods,
    datasets,
    hard_budget_ignored_datasets,
    soft_budget_ignored_datasets,
):
    """
    Compute which datasets are actually used for each method.

    A dataset is used by a method if it's not ignored by both hard and soft
    budgets. If a method ignores all budget types for a dataset, that dataset
    is considered unused for that method.

    Args:
        methods: List of method names
        datasets: List of dataset names
        hard_budget_ignored_datasets: Dict mapping method -> list of ignored datasets
        soft_budget_ignored_datasets: Dict mapping method -> list of ignored datasets

    Returns:
        Dict mapping method -> list of datasets used by that method
    """
    datasets_used = {}
    for method in methods:
        hard_ignored = set(hard_budget_ignored_datasets.get(method, []))
        soft_ignored = set(soft_budget_ignored_datasets.get(method, []))

        # A dataset is used if it's not ignored by both hard and soft budgets
        used_datasets = [
            dataset
            for dataset in datasets
            if not (dataset in hard_ignored and dataset in soft_ignored)
        ]
        datasets_used[method] = used_datasets

    return datasets_used
