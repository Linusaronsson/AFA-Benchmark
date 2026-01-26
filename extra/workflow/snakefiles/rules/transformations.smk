"""
Data transformation rules for evaluation results.

Handles transformations on evaluation results:
- Adding evaluation metadata (eval_soft_budget_param)
- Removing unnecessary columns (prev_selections_performed)
- Adding training metadata (method, dataset, seeds, budgets)
- Pivoting classifier columns to tidy data format
"""


rule transform_eval_data:
    """Transform raw evaluation data to final format for plotting.

    Applies all transformations in sequence:
    1. Add eval metadata (eval_soft_budget_param column)
    2. Remove selections history column (saves space)
    3. Add training metadata columns
    4. Pivot classifier columns to tidy data format
    """
    input:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results_transformed/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data_pipeline.py {input} {output} \
            {wildcards.method} \
            {wildcards.dataset} \
            {wildcards.train_seed} \
            {wildcards.train_hard_budget} \
            {wildcards.train_soft_budget_param}
        """
