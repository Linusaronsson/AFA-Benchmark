rule train_masked_mlp_classifier:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/trained_classifiers/masked_mlp_classifier/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "classifier_seed-{classifier_seed}.bundle"
        )
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset]
        script_name=lambda wildcards: CLASSIFIER_SCRIPT_NAMES[wildcards.dataset]
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
            seed={wildcards.classifier_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset}
        """
