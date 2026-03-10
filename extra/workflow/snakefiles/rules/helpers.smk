"""
Rules for running the pipeline up to a certain step.
"""

rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-builtin" for method_set in METHOD_SETS
        ] +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-external" for method_set in METHOD_SETS
        ] +
        # The next two sets of plots should be identical, but include them both just in case
        (
            [
                f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/method_set-{HEATMAP_METHOD_SET}+classifier_type-external"
            ]
            if HEATMAP_METHOD_SET in METHOD_SETS
            else []
        ) +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"
        ] +
        (
            [
                "extra/output/plot_results/cube_nm_ar/"
                f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
                f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/stop_shield-none"
            ]
            if "cube_nm_ar" in DATASETS
            else []
        ) +
        [
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
            f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/stop_shield-{delta}"
            for delta in STOP_SHIELD_DELTAS
            if "cube_nm_ar" in DATASETS
        ]

rule all_generate_dataset:
    input:
        [
            f"extra/output/datasets/{dataset}/{dataset_instance_idx}/{split}.bundle"
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for split in ["train", "val", "test"]
        ]

rule all_train_classifier:
    input:
        [
            (
                f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
                    f"dataset-{dataset}.bundle"
            )
            for dataset in DATASETS
        ] +
        [
            (
                f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
                    f"method-{method}+dataset-{dataset}.bundle"
            )
            for method in METHODS
            if method in METHOD_CLASSIFIER_SCRIPT_NAMES
            for dataset in DATASETS
        ]


rule all_pretrain_model:
    input:
        [
            (
                f"extra/output/pretrained_models/{TRAIN_INITIALIZER_TAG}/{pretrain_name}/"
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
                f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{method}/"
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
                f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{method}/"
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
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
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
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
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


rule all_eval_method_shielded:
    input:
        [
            (
                f"extra/output/eval_results_shielded/delta-{stop_shield_delta}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
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
            for stop_shield_delta in STOP_SHIELD_DELTAS
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results_shielded/delta-{stop_shield_delta}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
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
            for stop_shield_delta in STOP_SHIELD_DELTAS
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]
