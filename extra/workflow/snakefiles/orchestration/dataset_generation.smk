# List of datasets *has* to be provided
DATASETS = config["datasets"]

DATASET_INSTANCE_INDICES = config.get("dataset_instance_indices", list(range(5)))


rule all:
    input:
        expand("extra/output/datasets/{dataset}", dataset=DATASETS),


# Generate instances for a single type of dataset
# Use same seeds as instance indices
rule dataset_generation:
    output:
        directory("extra/output/datasets/{dataset}"),
    params:
        instance_indices_str=lambda wildcards: "["
        + ",".join(str(i) for i in DATASET_INSTANCE_INDICES)
        + "]",
        # Image datasets use a separate generation script because they are
        # defined by external files + transforms. We save only split indices
        # and config (not image tensors) to avoid large artifacts and freezing
        # augmentation behaviour.
        dataset_generation_script=lambda wildcards: (
            "generate_image_dataset.py"
            if wildcards.dataset
            in [
                "imagenette",
            ]
            else "generate_dataset.py"
        ),
    shell:
        """
        python scripts/dataset_generation/{params.dataset_generation_script} \
            dataset={wildcards.dataset} \
            instance_indices={params.instance_indices_str} \
            seeds={params.instance_indices_str} \
            save_path={output}
        """
