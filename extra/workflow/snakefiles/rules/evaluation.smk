"""
Evaluation rules.

Handles evaluation of trained methods on test/validation datasets.
"""


rule eval_method:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle",

        f"extra/output/trained_classifiers/"
            "dataset-{dataset}.bundle"

    output:
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
        "extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output[0]} \
            classifier_bundle_path={input[2]} \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget={wildcards.eval_hard_budget} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
