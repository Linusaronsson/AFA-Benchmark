"""
Evaluation rules.

Handles evaluation of trained methods on test/validation datasets.
"""


def _classifier_bundle_for_method(method: str, dataset: str) -> str:
    classifier_method = METHOD_TO_CLASSIFIER_BUNDLE_METHOD.get(
        method, method
    )
    if classifier_method in METHOD_CLASSIFIER_SCRIPT_NAMES:
        return (
            f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
            f"method-{classifier_method}+dataset-{dataset}.bundle"
        )
    return (
        f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
        f"dataset-{dataset}.bundle"
    )


rule eval_method:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle",

        lambda wildcards: _classifier_bundle_for_method(
            wildcards.method, wildcards.dataset
        )

    output:
        f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
        f"extra/output/eval_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
            components/initializers@initializer={EVAL_INITIALIZER} \
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


rule eval_method_shielded:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle",

        lambda wildcards: _classifier_bundle_for_method(
            wildcards.method, wildcards.dataset
        )

    output:
        f"extra/output/eval_results_shielded/delta-{{stop_shield_delta}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
        f"extra/output/eval_time_results_shielded/delta-{{stop_shield_delta}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
            components/initializers@initializer={EVAL_INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output[0]} \
            classifier_bundle_path={input[2]} \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget={wildcards.eval_hard_budget} \
            soft_budget_param={wildcards.eval_soft_budget_param} \
            batch_size={params.eval_batch_size} \
            stop_shield_delta={wildcards.stop_shield_delta} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule eval_method_dualized:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle",

        lambda wildcards: _classifier_bundle_for_method(
            wildcards.method, wildcards.dataset
        )

    output:
        f"extra/output/eval_results_dualized/lambda-{{dual_lambda}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
        f"extra/output/eval_time_results_dualized/lambda-{{dual_lambda}}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
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
            components/initializers@initializer={EVAL_INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output[0]} \
            classifier_bundle_path={input[2]} \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget={wildcards.eval_hard_budget} \
            soft_budget_param={wildcards.eval_soft_budget_param} \
            batch_size={params.eval_batch_size} \
            dual_lambda={wildcards.dual_lambda} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
