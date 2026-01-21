"""
Data transformation rules for evaluation results.

Handles sequential transformations on evaluation results:
- Adding evaluation metadata
- Removing unnecessary columns
- Adding training metadata
- Validating budget parameters
- Pivoting classifier columns to tidy data format
"""


rule add_eval_metadata_to_eval_data:
    """Add eval_soft_budget_param column. Not used by dummy methods, but kept for consistency."""
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
        "extra/output/eval_results2/{method}/"
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
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col eval_soft_budget_param=""
        """


rule remove_selections:
    """Remove the column containing the history of all selections (takes up a lot of space, unused for plots)."""
    input:
        "extra/output/eval_results2/{method}/"
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
        "extra/output/eval_results3/{method}/"
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
        shell_exec="nu"
    shell:
        """
        open {input} | reject prev_selections_performed | save {output}
        """


rule add_train_metadata_to_eval_data:
    """Add metadata columns from training (method, dataset, seeds, budgets)."""
    input:
        "extra/output/eval_results3/{method}/"
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
        "extra/output/eval_results4/{method}/"
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
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col afa_method={wildcards.method} \
            --col dataset={wildcards.dataset} \
            --col train_seed={wildcards.train_seed} \
            --col train_hard_budget={wildcards.train_hard_budget} \
            --col train_soft_budget_param={wildcards.train_soft_budget_param}
        """


rule validate_hard_budget_and_soft_budget_param:
    """
    Validate hard and soft budget parameters.

    Ensures:
    - train_hard_budget and eval_hard_budget are the same (rename to hard_budget)
    - Only one of train_soft_budget_param and eval_soft_budget_param is set
    - Rename to soft_budget_param
    """
    input:
        "extra/output/eval_results4/{method}/"
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
        "extra/output/eval_results5/{method}/"
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
        python scripts/misc/transform_eval_data.py validate_budgets {input} {output}
        """


rule pivot_long_classifier:
    """Convert classifier columns to tidy data format expected by plotting script."""
    input:
        "extra/output/eval_results5/{method}/"
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
        "extra/output/eval_results6/{method}/"
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
        python scripts/misc/transform_eval_data.py pivot_long_classifier {input} {output}
        """
