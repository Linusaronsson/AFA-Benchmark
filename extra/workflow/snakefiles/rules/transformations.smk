"""
Data transformation rules for evaluation results.

Handles transformations on evaluation results:
- Adding evaluation metadata (eval_soft_budget_param)
- Removing unnecessary columns (prev_selections_performed)
- Adding training metadata (method, dataset, seeds, budgets)
- Pivoting classifier columns to tidy data format
"""

from afabench.common.naming import resolve_existing_dataset_path


def _resolve_eval_csv_input(
    wildcards,
    *,
    root: str,
) -> str:
    canonical_path = (
        f"{root}/{wildcards.method}/"
        f"dataset-{wildcards.dataset}+"
        f"instance_idx-{wildcards.dataset_instance_idx}/"
        f"{wildcards.pretrain_folder}"
        f"train_seed-{wildcards.train_seed}+"
        f"train_hard_budget-{wildcards.train_hard_budget}+"
        f"train_soft_budget_param-{wildcards.train_soft_budget_param}/"
        f"eval_seed-{wildcards.eval_seed}+"
        f"eval_hard_budget-{wildcards.eval_hard_budget}+"
        f"eval_soft_budget_param-{wildcards.eval_soft_budget_param}/"
        "eval_data.csv"
    )
    return resolve_existing_dataset_path(
        canonical_path,
        wildcards.dataset,
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
        lambda wildcards: _resolve_eval_csv_input(
            wildcards,
            root=(
                "extra/output/eval_results/"
                f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}"
            ),
        ),
    output:
        f"extra/output/eval_results_transformed/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.parquet",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data_pipeline.py \
            --input_path {input} \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --initializer {INITIALIZER_TAG} \
            --train_seed {wildcards.train_seed} \
            --train_hard_budget {wildcards.train_hard_budget} \
            --train_soft_budget_param {wildcards.train_soft_budget_param} \
            --eval_soft_budget_param {wildcards.eval_soft_budget_param}
        """


rule transform_eval_data_shielded:
    """Transform threshold-shielded evaluation data for aggregation."""
    input:
        lambda wildcards: _resolve_eval_csv_input(
            wildcards,
            root=(
                "extra/output/eval_results_shielded/"
                f"delta-{wildcards.stop_shield_delta}/"
                f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}"
            ),
        ),
    output:
        f"extra/output/eval_results_transformed_shielded/delta-{{stop_shield_delta}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.parquet",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data_pipeline.py \
            --input_path {input} \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --initializer {INITIALIZER_TAG} \
            --train_seed {wildcards.train_seed} \
            --train_hard_budget {wildcards.train_hard_budget} \
            --train_soft_budget_param {wildcards.train_soft_budget_param} \
            --eval_soft_budget_param {wildcards.eval_soft_budget_param}
        """


rule transform_eval_data_dualized:
    """Transform dualized-stop evaluation data for aggregation."""
    input:
        lambda wildcards: _resolve_eval_csv_input(
            wildcards,
            root=(
                "extra/output/eval_results_dualized/"
                f"lambda-{wildcards.dual_lambda}/"
                f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}"
            ),
        ),
    output:
        f"extra/output/eval_results_transformed_dualized/lambda-{{dual_lambda}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.parquet",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data_pipeline.py \
            --input_path {input} \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --initializer {INITIALIZER_TAG} \
            --train_seed {wildcards.train_seed} \
            --train_hard_budget {wildcards.train_hard_budget} \
            --train_soft_budget_param {wildcards.train_soft_budget_param} \
            --eval_soft_budget_param {wildcards.eval_soft_budget_param}
        """
