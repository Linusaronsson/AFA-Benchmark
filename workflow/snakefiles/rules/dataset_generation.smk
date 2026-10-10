# Generate one dataset realization of a single type of dataset, one job per
# realization so that adding realizations leaves the existing ones untouched.
# Use the dataset realization index as seed
rule dataset_generation:
    output:
        [
            directory(
                OUTPUT_LAYOUT.dataset_bundle(
                    dataset="{dataset}",
                    dataset_realization_index="{dataset_realization_index}",
                    split=split,
                )
            )
            for split in ["train", "val", "test"]
        ],
        job_record=OUTPUT_LAYOUT.dataset_generation_job_record(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
        ),
    wildcard_constraints:
        dataset_realization_index=r"\d+",
    params:
        image=IMAGES.param("dataset_generation", lambda wc: wc.dataset),
        job_record=JOB_RECORDS.param("dataset_generation", lambda wc: wc.dataset),
        save_path=lambda wc: OUTPUT_LAYOUT.dataset_folder(dataset=wc.dataset),
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
        {params.image}{params.job_record} \
        python scripts/dataset_generation/{params.dataset_generation_script} \
            dataset={wildcards.dataset} \
            dataset_realization_indices=[{wildcards.dataset_realization_index}] \
            seeds=[{wildcards.dataset_realization_index}] \
            save_path={params.save_path}
        """
