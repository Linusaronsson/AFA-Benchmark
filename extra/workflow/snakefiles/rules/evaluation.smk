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
        f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_data.csv",
        f"extra/output/eval_time_results/eval_split-{EVAL_DATASET_SPLIT}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        eval_batch_size=lambda wildcards: EVAL_BATCH_SIZES[wildcards.method][wildcards.dataset],
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
            soft_budget_param={wildcards.eval_soft_budget_param} \
            batch_size={params.eval_batch_size} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
