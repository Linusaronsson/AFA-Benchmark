"""
Rules for running the pipeline up to a certain step.
"""

# The method set whose actions are also plotted as heatmaps.
HEATMAP_METHOD_SET = "heatmap_comparison"

rule all:
    input:
        [
            OUTPUT_LAYOUT.eval_perf_plots(
                method_set=method_set, classifier_type=classifier_type
            )
            for method_set in WORKFLOW_SETTINGS.method_sets
            for classifier_type in ["builtin", "external"]
        ]
        + [
            OUTPUT_LAYOUT.eval_actions_plots(
                method_set=method_set, classifier_type="external"
            )
            for method_set in WORKFLOW_SETTINGS.method_sets
            if method_set == HEATMAP_METHOD_SET
        ]
        + [OUTPUT_LAYOUT.time_plots(), OUTPUT_LAYOUT.job_duration_table()]

rule all_generate_datasets:
    input:
        [
            OUTPUT_LAYOUT.dataset_bundle(
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                split=split,
            )
            for dataset in WORKFLOW_SETTINGS.datasets
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
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
            for dataset in WORKFLOW_SETTINGS.datasets
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
        ] +
        [
            OUTPUT_LAYOUT.classifier_bundle(
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                method=method,
            )
            for method in WORKFLOW_SETTINGS.methods
            if method in WORKFLOW_SETTINGS.method_classifier_script_names
            for dataset in WORKFLOW_SETTINGS.datasets
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
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
            for pretrain_name in WORKFLOW_SETTINGS.pretrain_names
            for dataset in WORKFLOW_SETTINGS.datasets_used_per_pretrain_name[pretrain_name]
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
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
            for method in WORKFLOW_SETTINGS.methods
            for dataset in WORKFLOW_SETTINGS.datasets
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
            for budget_combination in WORKFLOW_SETTINGS.budget_params[method][dataset]
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
            for method in WORKFLOW_SETTINGS.methods
            for dataset in WORKFLOW_SETTINGS.datasets
            for dataset_realization_index in WORKFLOW_SETTINGS.dataset_realization_indices
            for budget_combination in WORKFLOW_SETTINGS.budget_params[method][dataset]
        ]
