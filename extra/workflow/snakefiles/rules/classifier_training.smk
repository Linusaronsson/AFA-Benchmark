# Every dataset realization gets its own classifiers, trained on its train and
# val splits with its index as seed, so no classifier sees the test split of
# the realization it is used on. See
# docs/adr/0005-classifiers-trained-per-dataset-realization.md.

from execution import checked_script_params


def _classifier_script_name(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return classifier_cfg["script_name"]
    return classifier_cfg


def _classifier_script_params(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return checked_script_params(
            " ".join(classifier_cfg.get("script_params", [])),
            f"classifier script_params for {dataset!r}",
        )
    return ""


def _method_classifier_script_name(method: str, dataset: str) -> str:
    script_name = METHOD_CLASSIFIER_SCRIPT_NAMES.get(method)
    if script_name is not None:
        return script_name
    return _classifier_script_name(dataset)


def _method_classifier_script_params(method: str, dataset: str) -> str:
    script_params = METHOD_CLASSIFIER_SCRIPT_PARAMS.get(method)
    if script_params is not None:
        return checked_script_params(
            script_params, f"classifier script_params for method {method!r}"
        )
    return _classifier_script_params(dataset)


rule train_classifier:
    input:
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split="train",
        ),
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split="val",
        ),
    output:
        directory(
            OUTPUT_LAYOUT.classifier_bundle(
                dataset="{dataset}",
                dataset_realization_index="{dataset_realization_index}",
                method=None,
            )
        )
    params:
        device=lambda wc, resources: EXECUTION.checked_device("classifier_training", None, resources),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: _classifier_script_name(
            wildcards.dataset
        ),
        script_params=lambda wildcards: _classifier_script_params(
            wildcards.dataset
        ),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("classifier_training", lambda wc: None),
    shell:
        """
        python scripts/train_classifier/{params.script_name}.py \
            train_dataset_path={input[0]} \
            val_dataset_path={input[1]} \
            save_path={output} \
            initializer={INITIALIZER} \
            unmasker={params.unmasker} \
            device={params.device} \
            seed={wildcards.dataset_realization_index} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.script_params}
        """


rule train_classifier_for_method:
    input:
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split="train",
        ),
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split="val",
        ),
    output:
        directory(
            OUTPUT_LAYOUT.classifier_bundle(
                dataset="{dataset}",
                dataset_realization_index="{dataset_realization_index}",
                method="{method}",
            )
        )
    params:
        device=lambda wc, resources: EXECUTION.checked_device("classifier_training", wc.method, resources),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: _method_classifier_script_name(
            wildcards.method, wildcards.dataset
        ),
        script_params=lambda wildcards: _method_classifier_script_params(
            wildcards.method, wildcards.dataset
        ),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("classifier_training", lambda wc: wc.method),
    shell:
        """
        python scripts/train_classifier/{params.script_name}.py \
            train_dataset_path={input[0]} \
            val_dataset_path={input[1]} \
            save_path={output} \
            initializer={INITIALIZER} \
            unmasker={params.unmasker} \
            device={params.device} \
            seed={wildcards.dataset_realization_index} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.script_params}
        """
