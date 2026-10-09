# Generate dataset realizations for a single type of dataset
# Use the dataset realization indices as seeds
rule dataset_generation:
    output:
        [
            directory(
                OUTPUT_LAYOUT.dataset_bundle(
                    dataset="{dataset}",
                    dataset_realization_index=dataset_realization_index,
                    split=split,
                )
            )
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for split in ["train", "val", "test"]
        ],
        job_record=OUTPUT_LAYOUT.dataset_generation_job_record(dataset="{dataset}"),
    params:
        job_record=JOB_RECORDS.param("dataset_generation", lambda wc: wc.dataset),
        save_path=lambda wc: OUTPUT_LAYOUT.dataset_folder(dataset=wc.dataset),
        dataset_realization_indices_str=lambda wildcards: "["
        + ",".join(str(i) for i in DATASET_REALIZATION_INDICES)
        + "]",
        # Image datasets use a separate generation script because they are
        # defined by external files + transforms. We save only generation
        # indices and config (not image tensors) to avoid large artifacts and
        # freezing augmentation behaviour.
        dataset_generation_script=lambda wildcards: (
            "generate_image_dataset.py"
            if wildcards.dataset
            in [
                "imagenette",
            ]
            else "generate_dataset.py"
        ),
    resources:
        **EXECUTION.allocation_resources("dataset_generation", lambda wc: wc.dataset),
    shell:
        """
        {params.job_record} \
        python scripts/dataset_generation/{params.dataset_generation_script} \
            dataset={wildcards.dataset} \
            dataset_realization_indices={params.dataset_realization_indices_str} \
            seeds={params.dataset_realization_indices_str} \
            save_path={params.save_path}
        """
