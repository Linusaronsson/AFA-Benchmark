"""
Data transformation rules for evaluation results.

Handles transformations on evaluation results:
- Adding evaluation metadata (eval_soft_budget_param)
- Removing unnecessary columns (prev_selections_performed)
- Adding training metadata (method, dataset, seeds, budgets)
- Pivoting classifier columns to tidy data format
"""

import re

# Reference methods' plotting-ready tables are restored from a benchmark
# release. Not matching them here leaves those tables as plain input files,
# so aggregating them never schedules their evaluation or training.
TRANSFORMED_METHOD_PATTERN = (
    "(?!(?:" + "|".join(map(re.escape, REFERENCE_METHODS)) + ")/).+"
    if REFERENCE_METHODS
    else ".+"
)


rule transform_eval_data:
    """Transform raw evaluation data to final format for plotting.

    Applies all transformations in sequence:
    1. Add eval metadata (eval_soft_budget_param column)
    2. Remove selections history column (saves space)
    3. Add training metadata columns
    4. Pivot classifier columns to tidy data format
    """
    input:
        f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "realization_index-{dataset_realization_index}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.parquet",
    output:
        f"extra/output/eval_results_transformed/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "realization_index-{dataset_realization_index}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.parquet",
    wildcard_constraints:
        method=TRANSFORMED_METHOD_PATTERN,
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("transformation", "transform_eval_data", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("transformation", lambda wc: "transform_eval_data"),
    shell:
        """
        python scripts/misc/transform_eval_data_pipeline.py \
            --input_path {input} \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --initializer {INITIALIZER} \
            --train_seed {wildcards.train_seed} \
            --train_hard_budget {wildcards.train_hard_budget} \
            --train_soft_budget_param {wildcards.train_soft_budget_param} \
            --eval_soft_budget_param {wildcards.eval_soft_budget_param}
        """
