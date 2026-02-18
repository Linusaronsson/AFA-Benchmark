# We only train the classifier once per dataset (on the first instance)


def _classifier_script_name(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return classifier_cfg["script_name"]
    return classifier_cfg


def _classifier_script_params(dataset: str) -> str:
    classifier_cfg = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier_cfg, dict):
        return " ".join(classifier_cfg.get("script_params", []))
    return ""


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
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: _classifier_script_name(
            wildcards.dataset
        ),
        script_params=lambda wildcards: _classifier_script_params(
            wildcards.dataset
        ),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/train/{params.script_name}.py \
            train_dataset_path={input[0]} \
            val_dataset_path={input[1]} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={DEVICE} \
            seed=0 \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset} \
            {params.script_params}
        """
