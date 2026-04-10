"""
Training rules for RL and dummy methods.

Handles:
- Pretraining of models
- Training with and without pretrained models
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


rule pretrain_model:
    input:
        # Datasets
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
        # Classifier
        ancient(f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
            "dataset-{dataset}.bundle")
    output:
        directory(
            f"extra/output/pretrained_models/{TRAIN_INITIALIZER_TAG}/{{pretrained_model_name}}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "model.bundle"
        ),

        f"extra/output/pretrained_models/{TRAIN_INITIALIZER_TAG}/{{pretrained_model_name}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "pretrain_time.txt"

    params:
        script_name=lambda wildcards: PRETRAIN_SCRIPT_NAMES[wildcards.pretrained_model_name],
        pretrain_params=lambda wildcards: PRETRAIN_PARAMS[wildcards.pretrained_model_name],
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        device=lambda wildcards: _pretrain_device(
            wildcards.pretrained_model_name
        )
    resources:
        shell_exec="bash",
        slurm_extra=lambda wildcards: _pretrain_slurm_extra(
            wildcards.pretrained_model_name
        )
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/pretrain/{params.script_name}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            classifier_bundle_path={input[2]} \
            save_path={output[0]} \
            components/initializers@initializer={TRAIN_INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} \
            seed={wildcards.pretrain_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.pretrain_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule train_method_with_pretrained_model:
    input:
        # Datasets
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        # Pretrained model
        lambda wildcards: (
            f"extra/output/pretrained_models/{TRAIN_INITIALIZER_TAG}/{METHOD_TO_PRETRAINED_MODEL[wildcards.method]}/"
                f"dataset-{wildcards.dataset}+"
                f"instance_idx-{wildcards.dataset_instance_idx}/"
                    f"pretrain_seed-{wildcards.pretrain_seed}/"
                        "model.bundle"
        ),

        # Classifier
        ancient(
            f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
            "dataset-{dataset}.bundle"
        )

    output:
        directory(
            f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: METHOD_TRAIN_SCRIPT_NAMES[wildcards.method],
        method_specific_params=lambda wildcards: METHOD_SPECIFIC_PARAMS[wildcards.method],
        device=lambda wildcards: _method_device(wildcards.method)
    resources:
        shell_exec="bash",
        slurm_extra=lambda wildcards: _method_slurm_extra(
            wildcards.method
        )
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{params.script_name}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            pretrained_model_bundle_path={input[2]} \
            classifier_bundle_path={input[3]} \
            save_path={output[0]} \
            components/initializers@initializer={TRAIN_INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={params.device} \
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
        # Datasets
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        # Classifier
        ancient(
            lambda wildcards: _classifier_bundle_for_method(
                wildcards.method, wildcards.dataset
            )
        )
    output:
        directory(
            f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: METHOD_TRAIN_SCRIPT_NAMES[wildcards.method],
        method_specific_params=lambda wildcards: METHOD_SPECIFIC_PARAMS[wildcards.method],
        device=lambda wildcards: _method_device(wildcards.method)
    resources:
        shell_exec="bash",
        slurm_extra=lambda wildcards: _method_slurm_extra(
            wildcards.method
        )
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{params.script_name}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            classifier_bundle_path={input[2]} \
            save_path={output[0]} \
            components/initializers@initializer={TRAIN_INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={params.device} \
            seed={wildcards.train_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.method_specific_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
