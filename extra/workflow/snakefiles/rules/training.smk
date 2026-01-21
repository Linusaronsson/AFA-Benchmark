"""
Training rules for RL and dummy methods.

Handles:
- Pretraining of models
- Training with and without pretrained models
"""


rule pretrain_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/pretrained_models/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "model.bundle"
        ),

        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "pretrain_time.txt"

    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/pretrain/{wildcards.method}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            device={DEVICE} \
            seed={wildcards.pretrain_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule train_method_with_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "model.bundle"
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: METHOD_TO_SCRIPT_NAME_MAPPING[wildcards.method]
        method_specific_params=lambda wildcards: METHOD_SPECIFIC_PARAMS[wildcards.method]
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{params.script_name}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            pretrained_model_bundle_path={input[2]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.method_specific_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule train_method_without_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: METHOD_TO_SCRIPT_NAME_MAPPING[wildcards.method]
        method_specific_params=lambda wildcards: METHOD_SPECIFIC_PARAMS[wildcards.method]
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{params.script_name}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.method_specific_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
