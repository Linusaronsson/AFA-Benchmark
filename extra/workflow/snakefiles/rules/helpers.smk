"""
Rules for running the pipeline up to a certain step.
"""

rule all:
    input:
        [
            f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-builtin" for method_set in METHOD_SETS
        ] +
        [
            f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-external" for method_set in METHOD_SETS
        ] +
        # The next two sets of plots should be identical, but include them both just in case
        (
            [
                f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/method_set-{HEATMAP_METHOD_SET}+classifier_type-external"
            ]
            if HEATMAP_METHOD_SET in METHOD_SETS
            else []
        ) +
        [
            f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"
        ] +
        [
            f"{OUTPUT_ROOT}/merged_results/job_duration_table.parquet"
        ]

rule all_generate_datasets:
    input:
        [
            OUTPUT_LAYOUT.dataset_bundle(
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                split=split,
            )
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for split in ["train", "val", "test"]
        ]


rule all_train_classifiers:
    input:
        [
            OUTPUT_LAYOUT.classifier_bundle(
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                method=None,
            )
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
        ] +
        [
            OUTPUT_LAYOUT.classifier_bundle(
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                method=method,
            )
            for method in METHODS
            if method in METHOD_CLASSIFIER_SCRIPT_NAMES
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
        ]


rule all_pretrain_models:
    input:
        [
            OUTPUT_LAYOUT.pretrained_model_bundle(
                pretrained_model_name=pretrain_name,
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                pretrain_seed=dataset_realization_index,
            )
            for pretrain_name in PRETRAIN_NAMES
            for dataset in DATASETS_USED_PER_PRETRAIN_NAME[pretrain_name]
            for dataset_realization_index in DATASET_REALIZATION_INDICES
        ]


rule all_train_methods:
    input:
        [
            OUTPUT_LAYOUT.method_bundle(
                WORKFLOW_SETTINGS.evaluation_run(
                    method=method,
                    dataset=dataset,
                    dataset_realization_index=dataset_realization_index,
                    budget_combination=budget_combination,
                ).training
            )
            for method in METHODS
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for budget_combination in BUDGET_PARAMS[method][dataset]
        ]

rule all_eval_methods:
    input:
        [
            OUTPUT_LAYOUT.raw_evaluation_table(
                WORKFLOW_SETTINGS.evaluation_run(
                    method=method,
                    dataset=dataset,
                    dataset_realization_index=dataset_realization_index,
                    budget_combination=budget_combination,
                )
            )
            for method in METHODS
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for budget_combination in BUDGET_PARAMS[method][dataset]
        ]
