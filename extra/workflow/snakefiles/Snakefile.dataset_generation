DATASETS = [
    "afa_context",
    "afa_context_without_noise",
    # "fashion_mnist", # unexpected keyword argument 'load_subdirs'
    # "mnist", # unexpected keyword argument 'load_subdirs'
    "synthetic_mnist",
    "synthetic_mnist_without_noise",
    "actg",
    "bank_marketing",
    "ckd",
    "cube",
    "cube_without_noise",
    "diabetes",
    "imagenette",
    "miniboone",
    "physionet",
]
NUM_INSTANCES = 5
INSTANCE_INDICES = range(NUM_INSTANCES)


rule all:
    input:
        expand("extra/data/datasets/{dataset}", dataset=DATASETS),


# Generate instances for a single type of dataset
# Use same seeds as instance indices
rule dataset_generation:
    output:
        directory("extra/data/datasets/{dataset}"),
    params:
        instance_indices_str=lambda wildcards: "["
        + ",".join(str(i) for i in INSTANCE_INDICES)
        + "]",
        dataset_generation_script=lambda wildcards: (
            "generate_image_dataset.py"
            if wildcards.dataset
            in [
                "mnist",
                "fashion_mnist",
                "imagenette",
            ]
            else "generate_dataset.py"
        ),
    shell:
        """
        python scripts/dataset_generation/{params.dataset_generation_script} \
            scripts/dataset_generation/dataset@dataset={wildcards.dataset} \
            instance_indices={params.instance_indices_str} \
            seeds={params.instance_indices_str} \
            save_path={output}
        """
