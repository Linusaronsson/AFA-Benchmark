# We only train the classifier once per dataset (on the first instance)

from prerequisite_execution import checked_prerequisite_params


def _classifier_script_name(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return classifier_cfg["script_name"]
    return classifier_cfg


def _classifier_script_params(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return checked_prerequisite_params(
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
        return checked_prerequisite_params(
            script_params, f"classifier script_params for method {method!r}"
        )
    return _classifier_script_params(dataset)


rule train_classifier:
    input:
        "extra/output/datasets/{dataset}/0/train.bundle",
        "extra/output/datasets/{dataset}/0/val.bundle"
    output:
        directory(
            f"extra/output/trained_classifiers/{INITIALIZER_TAG}/"
                "dataset-{dataset}.bundle"
        )
    params:
        device=lambda wc, resources: EXECUTION.checked_device("classifier", None, resources),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: _classifier_script_name(
            wildcards.dataset
        ),
        script_params=lambda wildcards: _classifier_script_params(
            wildcards.dataset
        ),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "classifier", None)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "classifier", None)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "classifier", None),
        gres=lambda wc: EXECUTION.resource("gres", "classifier", None),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "classifier", None),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "classifier", None),
    shell:
        """
        python scripts/train_classifier/{params.script_name}.py \
            train_dataset_path={input[0]} \
            val_dataset_path={input[1]} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} \
            seed=0 \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.script_params}
        """


rule train_classifier_for_method:
    input:
        "extra/output/datasets/{dataset}/0/train.bundle",
        "extra/output/datasets/{dataset}/0/val.bundle"
    output:
        directory(
            f"extra/output/trained_classifiers/{INITIALIZER_TAG}/"
            "method-{method}+dataset-{dataset}.bundle"
        )
    params:
        device=lambda wc, resources: EXECUTION.checked_device("classifier", wc.method, resources),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: _method_classifier_script_name(
            wildcards.method, wildcards.dataset
        ),
        script_params=lambda wildcards: _method_classifier_script_params(
            wildcards.method, wildcards.dataset
        ),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "classifier", wc.method)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "classifier", wc.method)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "classifier", wc.method),
        gres=lambda wc: EXECUTION.resource("gres", "classifier", wc.method),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "classifier", wc.method),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "classifier", wc.method),
    shell:
        """
        python scripts/train_classifier/{params.script_name}.py \
            train_dataset_path={input[0]} \
            val_dataset_path={input[1]} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} \
            seed=0 \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.script_params}
        """
