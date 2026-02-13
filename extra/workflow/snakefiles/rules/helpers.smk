"""
Rules for running the pipeline up to a certain step.
"""

rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/eval_perf/method_set-{method_set}+classifier_type-builtin" for method_set in METHOD_SETS
        ] +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/eval_perf/method_set-{method_set}+classifier_type-external" for method_set in METHOD_SETS
        ] +
        # The next two sets of plots should be identical, but include them both just in case
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/eval_actions/method_set-{HEATMAP_METHOD_SET}+classifier_type-external"
        ] +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/time/"
        ]

rule all_generate_dataset:
    input:
        expand("extra/output/datasets/{dataset}", dataset=DATASETS),

rule all_train_classifier:
    input:
        [
            (
                "extra/output/trained_classifiers/"
                    f"dataset-{dataset}.bundle"
            )
            for dataset in DATASETS
        ]


rule all_pretrain_model:
    input:
        [
            (
                f"extra/output/pretrained_models/{pretrain_name}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            "model.bundle"
            )
            for pretrain_name in PRETRAIN_NAMES
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS_USED_PER_METHOD[method]
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
        ]


rule all_train_method:
    input:
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                "method.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, _eval_hard_budget, train_soft_budget_param, _eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                "method.bundle"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, _eval_hard_budget, train_soft_budget_param, _eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]

rule all_eval_method:
    input:
        [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]
