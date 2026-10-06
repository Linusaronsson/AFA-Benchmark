# Generate instances for a single type of dataset
# Use same seeds as instance indices
rule dataset_generation:
    output:
        [
            directory(f"extra/output/datasets/{{dataset}}/{dataset_instance_idx}/{split}.bundle") for dataset_instance_idx in DATASET_INSTANCE_INDICES for split in ["train", "val", "test"]
        ]
    params:
        # Validate final resources during planning; this script has no device argument.
        execution_device=lambda wc, resources: EXECUTION.checked_device("dataset_generation", wc.dataset, resources),
        save_path=lambda wc: f"extra/output/datasets/{wc.dataset}",
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
    resources:
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "dataset_generation", wc.dataset)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "dataset_generation", wc.dataset)) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "dataset_generation", wc.dataset),
        gres=lambda wc: EXECUTION.resource("gres", "dataset_generation", wc.dataset),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "dataset_generation", wc.dataset),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "dataset_generation", wc.dataset),
    shell:
        """
        python scripts/dataset_generation/{params.dataset_generation_script} \
            dataset={wildcards.dataset} \
            instance_indices={params.instance_indices_str} \
            seeds={params.instance_indices_str} \
            save_path={params.save_path}
        """
