"""
Training rules: the pretraining and training stages.

Both rules pass their script the training contract, rendered by
`extra/workflow/src/contract_arguments.py` as plain `key=value` arguments
(`docs/adr/0001-training-contract-as-library.md`), followed by the pretrained
model's `pretrain_params` or the method's `method_specific_params`.

`{pretrain_folder}` in `train_method` is `pretrain_seed-<seed>/` for methods
with a pretraining stage and `NO_PRETRAIN/` for the others, the same folder
the evaluation rules use, so the two former training rules are one.
"""

from execution import checked_script_params
from contract_arguments import (
    render_pretraining_contract,
    render_training_contract,
)


def _classifier_bundle_for_method(method: str, dataset: str) -> str:
    if method in METHOD_CLASSIFIER_SCRIPT_NAMES:
        return (
            f"extra/output/trained_classifiers/{INITIALIZER_TAG}/"
            f"method-{method}+dataset-{dataset}.bundle"
        )
    return (
        f"extra/output/trained_classifiers/{INITIALIZER_TAG}/"
        f"dataset-{dataset}.bundle"
    )


def _pretrained_model_bundle(wildcards) -> list[str]:
    """The pretrained model a training run needs: one bundle or none."""
    has_pretraining_stage = wildcards.method in METHOD_TO_PRETRAINED_MODEL
    if wildcards.pretrain_folder == f"{NO_PRETRAIN_STR}/":
        if has_pretraining_stage:
            raise ValueError(
                f"Method {wildcards.method!r} has a pretraining stage, so its "
                f"bundles live under pretrain_seed-<seed>/, not "
                f"{NO_PRETRAIN_STR}/."
            )
        return []
    if not has_pretraining_stage:
        raise ValueError(
            f"Method {wildcards.method!r} has no pretraining stage, so its "
            f"bundles live under {NO_PRETRAIN_STR}/, not "
            f"{wildcards.pretrain_folder}."
        )
    pretrain_seed = wildcards.pretrain_folder.removeprefix(
        "pretrain_seed-"
    ).removesuffix("/")
    return [
        f"extra/output/pretrained_models/{INITIALIZER_TAG}/"
        f"{METHOD_TO_PRETRAINED_MODEL[wildcards.method]}/"
        f"dataset-{wildcards.dataset}+"
        f"realization_index-{wildcards.dataset_realization_index}/"
        f"pretrain_seed-{pretrain_seed}/"
        "model.bundle"
    ]


def _pretraining_contract(wildcards, input, output, resources) -> str:
    return render_pretraining_contract(
        {
            "train_dataset_bundle_path": input.train_dataset,
            "val_dataset_bundle_path": input.val_dataset,
            "classifier_bundle_path": input.classifier,
            "save_path": output.model_bundle,
            "initializer": INITIALIZER,
            "unmasker": UNMASKERS[wildcards.dataset],
            "dataset_key": wildcards.dataset,
            "device": EXECUTION.checked_device("pretraining", wildcards.pretrained_model_name, resources),
            "seed": wildcards.pretrain_seed,
            "use_wandb": USE_WANDB,
            "smoke_test": SMOKE_TEST,
        }
    )


def _training_contract(wildcards, input, output, resources) -> str:
    return render_training_contract(
        {
            "train_dataset_bundle_path": input.train_dataset,
            "val_dataset_bundle_path": input.val_dataset,
            "classifier_bundle_path": input.classifier,
            "pretrained_model_bundle_path": (
                input.pretrained_model[0] if input.pretrained_model else None
            ),
            "save_path": output.method_bundle,
            "method_name": wildcards.method,
            "initializer": INITIALIZER,
            "unmasker": UNMASKERS[wildcards.dataset],
            "dataset_key": wildcards.dataset,
            "hard_budget": wildcards.train_hard_budget,
            "soft_budget_param": wildcards.train_soft_budget_param,
            "device": EXECUTION.checked_device("training", wildcards.method, resources),
            "seed": wildcards.train_seed,
            "use_wandb": USE_WANDB,
            "smoke_test": SMOKE_TEST,
        }
    )


rule pretrain_model:
    input:
        train_dataset="extra/output/datasets/{dataset}/{dataset_realization_index}/train.bundle",
        val_dataset="extra/output/datasets/{dataset}/{dataset_realization_index}/val.bundle",
        classifier=ancient(
            f"extra/output/trained_classifiers/{INITIALIZER_TAG}/"
            "dataset-{dataset}.bundle"
        ),
    output:
        model_bundle=directory(
            f"extra/output/pretrained_models/{INITIALIZER_TAG}/{{pretrained_model_name}}/"
                "dataset-{dataset}+"
                "realization_index-{dataset_realization_index}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "model.bundle"
        ),
        pretrain_time=(
            f"extra/output/pretrained_models/{INITIALIZER_TAG}/{{pretrained_model_name}}/"
                "dataset-{dataset}+"
                "realization_index-{dataset_realization_index}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "pretrain_time.txt"
        ),
    params:
        script_name=lambda wildcards: PRETRAIN_SCRIPT_NAMES[wildcards.pretrained_model_name],
        contract=_pretraining_contract,
        pretrain_params=lambda wildcards: checked_script_params(PRETRAIN_PARAMS[wildcards.pretrained_model_name], f"pretrain_params for {wildcards.pretrained_model_name!r}"),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("pretraining", lambda wc: wc.pretrained_model_name),
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/pretrain_model/{params.script_name}.py \
            {params.contract} \
            {params.pretrain_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output.pretrain_time}'
        """


rule train_method:
    input:
        train_dataset="extra/output/datasets/{dataset}/{dataset_realization_index}/train.bundle",
        val_dataset="extra/output/datasets/{dataset}/{dataset_realization_index}/val.bundle",
        pretrained_model=_pretrained_model_bundle,
        classifier=ancient(
            lambda wildcards: _classifier_bundle_for_method(
                wildcards.method, wildcards.dataset
            )
        ),
    output:
        method_bundle=directory(
            f"extra/output/trained_methods/{INITIALIZER_TAG}/{{method}}/"
                "dataset-{dataset}+"
                "realization_index-{dataset_realization_index}/"
                    "{pretrain_folder}"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        train_time=(
            f"extra/output/trained_methods/{INITIALIZER_TAG}/{{method}}/"
                "dataset-{dataset}+"
                "realization_index-{dataset_realization_index}/"
                    "{pretrain_folder}"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "train_time.txt"
        ),
    wildcard_constraints:
        pretrain_folder=rf"pretrain_seed-\d+/|{NO_PRETRAIN_STR}/",
    params:
        script_name=lambda wildcards: METHOD_TRAIN_SCRIPT_NAMES[wildcards.method],
        contract=_training_contract,
        method_specific_params=lambda wildcards: METHOD_SPECIFIC_PARAMS[wildcards.method],
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("training", lambda wc: wc.method),
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train_method/{params.script_name}.py \
            {params.contract} \
            {params.method_specific_params}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output.train_time}'
        """
