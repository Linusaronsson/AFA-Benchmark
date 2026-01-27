# We only train the classifier once per dataset (on the first instance)
rule train_classifier:
    input:
        "extra/output/datasets/{dataset}/0/train.bundle",
        "extra/output/datasets/{dataset}/0/val.bundle"
    output:
        directory(
            f"extra/output/trained_classifiers/"
                "dataset-{dataset}.bundle"
        )
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script_name=lambda wildcards: CLASSIFIER_NAMES[wildcards.dataset]
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
            experiment@_global_={wildcards.dataset}
        """
