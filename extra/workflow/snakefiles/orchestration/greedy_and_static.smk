import os
import time
from datetime import datetime

NO_PRETRAIN_STR = "NO_PRETRAIN"

DATASET_INSTANCE_INDICES = config.get("dataset_instance_indices", (0, 1))
INITIALIZER = config.get("initializer", "cold")
EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")
DEVICE = config.get("device", "cpu")

METHOD_OPTIONS = config.get("method_options", None)
if METHOD_OPTIONS is None:
    raise ValueError("Expected method_options to be provided.")
METHODS = config.get("methods", [])
if METHODS is None:
    raise ValueError("Expected methods to be provided.")
METHODS_WITH_PRETRAINING_STAGE = [
    method
    for method, options in METHOD_OPTIONS.items()
    if options["has_pretraining_stage"] and method in METHODS
]
METHODS_WITHOUT_PRETRAINING_STAGE = [
    method
    for method, options in METHOD_OPTIONS.items()
    if not options["has_pretraining_stage"] and method in METHODS
]
PRETRAIN_PROVIDER_BY_METHOD = {
    "ma2018_external": "zannone2019",
    "ma2018_builtin": "zannone2019",
}

METHODS_USING_FOREIGN_PRETRAIN = [
    m for m in METHODS
    if m in PRETRAIN_PROVIDER_BY_METHOD
]

PROVIDER_METHODS_TO_PRETRAIN = sorted({
    PRETRAIN_PROVIDER_BY_METHOD[m] for m in METHODS_USING_FOREIGN_PRETRAIN
})
ruleorder: train_method_with_foreign_pretrained_model > train_method_without_pretrained_model

DATASETS = config.get("datasets", None)
if DATASETS is None:
    raise ValueError("Expected datasets to be provided.")

UNMASKERS_RAW = config.get("unmaskers", None)
if UNMASKERS_RAW is None:
    raise ValueError("Expected unmaskers to be provided.")
UNMASKERS = {dataset: UNMASKERS_RAW["default"] for dataset in DATASETS} | UNMASKERS_RAW

HARD_BUDGETS_RAW = config.get("hard_budgets", None)
if HARD_BUDGETS_RAW is None:
    raise ValueError("Expected hard_budgets to be provided.")
HARD_BUDGETS = {dataset: HARD_BUDGETS_RAW["default"] for dataset in DATASETS} | HARD_BUDGETS_RAW

EVAL_HARD_BUDGETS_RAW = config.get("eval_hard_budgets", None)
if EVAL_HARD_BUDGETS_RAW is None:
    raise ValueError("Expected eval_hard_budgets to be provided.")
EVAL_HARD_BUDGETS = {dataset: EVAL_HARD_BUDGETS_RAW["default"] for dataset in DATASETS} | EVAL_HARD_BUDGETS_RAW

TRAIN_TO_EVAL_HARD_BUDGET_MAP_RAW = config.get("train_to_eval_hard_budget_map", None)
if TRAIN_TO_EVAL_HARD_BUDGET_MAP_RAW is None:
    raise ValueError("Expected train_to_eval_hard_budget_map to be provided.")
TRAIN_TO_EVAL_HARD_BUDGET_MAP = (
    TRAIN_TO_EVAL_HARD_BUDGET_MAP_RAW
    | {dataset: TRAIN_TO_EVAL_HARD_BUDGET_MAP_RAW.get("default", {}) for dataset in DATASETS if dataset not in TRAIN_TO_EVAL_HARD_BUDGET_MAP_RAW}
)

HARD_BUDGETS_PER_METHOD = {
    method: {
        dataset: list(HARD_BUDGETS[dataset])
        for dataset in DATASETS
    }
    for method in METHODS
}

EVAL_HARD_BUDGETS_PER_METHOD = {
    method: {
        dataset: list(EVAL_HARD_BUDGETS[dataset])
        for dataset in DATASETS
    }
    for method in METHODS
}

def _pairs_for_dataset(dataset: str, train_budgets: list[int], eval_budgets: list[int]) -> list[tuple[int, int]]:
    m = TRAIN_TO_EVAL_HARD_BUDGET_MAP.get(dataset, {})
    pairs: list[tuple[int, int]] = []
    train_set = {int(x) for x in train_budgets}
    for t_str, eval_list in m.items():
        t = int(t_str)
        if t not in train_set:
            raise ValueError(
                f"train_to_eval_hard_budget_map[{dataset}] references train budget {t} "
                f"but hard_budgets[{dataset}]={sorted(train_set)}"
            )
        for e in eval_list:
            pairs.append((t, int(e)))
    covered_eval = {e for _, e in pairs}
    missing = [int(e) for e in eval_budgets if int(e) not in covered_eval]
    if missing:
        raise ValueError(
            f"Eval budgets {missing} for dataset={dataset} have no mapping. "
            f"Add them to train_to_eval_hard_budget_map[{dataset}]."
        )
    return pairs

HARD_BUDGET_EVAL_PAIRS_PER_METHOD = {
    method: {
        dataset: _pairs_for_dataset(
            dataset,
            HARD_BUDGETS_PER_METHOD[method][dataset],
            EVAL_HARD_BUDGETS_PER_METHOD[method][dataset],
        )
        for dataset in DATASETS
    }
    for method in METHODS
}

SOFT_BUDGET_PARAMS_RAW = config.get("soft_budget_params", None)
if SOFT_BUDGET_PARAMS_RAW is None:
    raise ValueError("Expected soft_budget_params to be provided.")
SOFT_BUDGET_PARAMS = {
    method: (
        method_soft_budget_params
        | {dataset: method_soft_budget_params["default"] for dataset in DATASETS if dataset not in method_soft_budget_params}
    )
    for method, method_soft_budget_params in SOFT_BUDGET_PARAMS_RAW.items()
}
SOFT_BUDGET_PARAMS_PER_METHOD = {
    method: {
        dataset: list(SOFT_BUDGET_PARAMS[method][dataset])
        for dataset in DATASETS
    }
    for method in METHODS
}

MAX_HARD_BUDGET_PER_METHOD_DATASET = {
    method: {
        dataset: max(HARD_BUDGETS_PER_METHOD[method][dataset])
        for dataset in DATASETS
    }
    for method in METHODS
}

wildcard_constraints:
    dataset_instance_idx = r"\d+",
    pretrain_seed        = r"\d+",
    train_seed           = r"\d+",
    eval_seed            = r"\d+",
    train_hard_budget    = r"\d+",
    base_train_hard_budget = r"\d+",
    eval_hard_budget     = r"(?:null|\d+)",
    train_soft_budget_param = r"(?:null|[0-9]+(?:\.[0-9]+)?)"

def pretrain_script(wildcards):
    # Use *_image.py only for imagenette
    suffix = "_image" if wildcards.dataset == "imagenette" else ""
    return f"scripts/pretrain/{wildcards.method}{suffix}.py"

def train_script(wildcards):
    suffix = "_image" if wildcards.dataset == "imagenette" else ""
    base_method = "ma2018" if wildcards.method in {"ma2018_external", "ma2018_builtin"} else wildcards.method
    return f"scripts/train/{base_method}{suffix}.py"

rule all:
    input:
        "extra/output/plot_results/Greedy_and_static",


rule pretrain_model_all:
    input:
        [
            (
                f"extra/output/pretrained_models/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"model.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
        ]


rule train_method_all:
    input:
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            "train_soft_budget_param-null/"
                                "method.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for hard_budget in HARD_BUDGETS_PER_METHOD[method][dataset]
        ] +
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            "train_soft_budget_param-null/"
                                "method.bundle"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for hard_budget in HARD_BUDGETS_PER_METHOD[method][dataset]
        ]


rule pretrain_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/pretrained_models/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "model.bundle"
        ),
        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "pretrain_time.txt",
    params:
        script=pretrain_script,
    resources:
        shell_exec="bash",
    shell:
        """
        START_TIME=$(date +%s.%N)
        python {params.script} \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            device={DEVICE} \
            seed={wildcards.pretrain_seed} \
            experiment@_global_={wildcards.dataset}

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule train_method_with_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "model.bundle"
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-null/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-null/"
                        "train_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script=train_script,
    resources:
        shell_exec="bash",
    shell:
        """
        START_TIME=$(date +%s.%N)
        python {params.script} \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            pretrained_model_bundle_path={input[2]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            experiment@_global_={wildcards.dataset}

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """


rule train_method_with_foreign_pretrained_model:
    input:
        train_dataset="extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        val_dataset="extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        # Provider pretrained bundle, e.g. zannone2019 for ma2018
        pretrained_model=lambda wildcards: (
            "extra/output/pretrained_models/"
            f"{PRETRAIN_PROVIDER_BY_METHOD[wildcards.method]}/"
            f"dataset-{wildcards.dataset}+"
            f"instance_idx-{wildcards.dataset_instance_idx}/"
            f"pretrain_seed-{wildcards.dataset_instance_idx}/"
            f"model.bundle"
        ),
        provider_pretrain_time=lambda wildcards: (
            "extra/output/pretrained_models/"
            f"{PRETRAIN_PROVIDER_BY_METHOD[wildcards.method]}/"
            f"dataset-{wildcards.dataset}+"
            f"instance_idx-{wildcards.dataset_instance_idx}/"
            f"pretrain_seed-{wildcards.dataset_instance_idx}/"
            "pretrain_time.txt"
        ),
        trained_classifier=lambda wildcards: (
            f"extra/output/classifiers/masked_mlp_classifier/"
            f"dataset-{wildcards.dataset}+"
            f"instance_idx-{wildcards.dataset_instance_idx}/"
            f"seed-{wildcards.dataset_instance_idx}.bundle"
        ),
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-null/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-null/"
                        "train_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script=train_script,
    resources:
        shell_exec="bash",
    wildcard_constraints:
        method="|".join(METHODS_USING_FOREIGN_PRETRAIN) if METHODS_USING_FOREIGN_PRETRAIN else "a^",
    shell:
        r"""
        CLASSIFIER_ARG=""
        if [ "{wildcards.method}" = "ma2018_external" ]; then
            CLASSIFIER_ARG="trained_classifier_bundle_path={input.trained_classifier}"
        fi

        python {params.script} \
            train_dataset_bundle_path={input.train_dataset} \
            val_dataset_bundle_path={input.val_dataset} \
            pretrained_model_bundle_path={input.pretrained_model} \
            $CLASSIFIER_ARG \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            experiment@_global_={wildcards.dataset}

        cp "{input.provider_pretrain_time}" "{output[1]}"
        """


rule train_method_without_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-null/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-null/"
                        "train_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        script=train_script,
    wildcard_constraints:
        method="|".join([m for m in METHODS_WITHOUT_PRETRAINING_STAGE if m not in PRETRAIN_PROVIDER_BY_METHOD]) if METHODS_WITHOUT_PRETRAINING_STAGE else "a^",
    resources:
        shell_exec="bash",
    shell:
        """
        START_TIME=$(date +%s.%N)
        python {params.script} \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            experiment@_global_={wildcards.dataset}

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """

# resave only for the maximum hard budget
def base_model_for_soft(wc):
    maxb = MAX_HARD_BUDGET_PER_METHOD_DATASET[wc.method][wc.dataset]
    if int(wc.train_hard_budget) != int(maxb):
        raise ValueError(
            f"Soft-budget resave requested for non-max hard budget "
            f"{wc.train_hard_budget} (max is {maxb})"
        )
    return (
        f"extra/output/trained_methods/{wc.method}/"
        f"dataset-{wc.dataset}+"
        f"instance_idx-{wc.dataset_instance_idx}/"
        f"{wc.pretrain_folder}"
        f"train_seed-{wc.train_seed}+"
        f"train_hard_budget-{maxb}+"
        "train_soft_budget_param-null/"
        "method.bundle"
    )

def base_train_time_for_soft(wc):
    maxb = MAX_HARD_BUDGET_PER_METHOD_DATASET[wc.method][wc.dataset]
    return (
        f"extra/output/trained_methods/{wc.method}/"
        f"dataset-{wc.dataset}+"
        f"instance_idx-{wc.dataset_instance_idx}/"
        f"{wc.pretrain_folder}"
        f"train_seed-{wc.train_seed}+"
        f"train_hard_budget-{maxb}+"
        "train_soft_budget_param-null/"
        "train_time.txt"
    )

rule resave_with_soft_budget:
    input:
        base_model=base_model_for_soft,
        base_time=base_train_time_for_soft,
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "{pretrain_folder}"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                    "train_time.txt",
    wildcard_constraints:
        train_soft_budget_param=r"[0-9]+(?:\.[0-9]+)?"
    resources:
        shell_exec="bash",
    shell:
        """
        python scripts/utils/resave_with_soft_budget.py \
            trained_model_bundle_path={input.base_model} \
            save_path={output[0]} \
            device={DEVICE} \
            soft_budget_param={wildcards.train_soft_budget_param}

        cp "{input.base_time}" "{output[1]}"
        """

def train_hard_budget_for_eval(wc) -> int:
    pairs = HARD_BUDGET_EVAL_PAIRS_PER_METHOD[wc.method][wc.dataset]
    e = int(wc.eval_hard_budget)
    matches = [t for (t, ee) in pairs if ee == e]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one train budget for method={wc.method} dataset={wc.dataset} eval={e}, "
            f"got {matches}. Fix train_to_eval_hard_budget_map to be one-to-one per eval budget."
        )
    return matches[0]

rule eval_method_hard:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        lambda wc: (
            f"extra/output/trained_methods/{wc.method}/"
            f"dataset-{wc.dataset}+"
            f"instance_idx-{wc.dataset_instance_idx}/"
            f"{wc.pretrain_folder}"
            f"train_seed-{wc.train_seed}+"
            f"train_hard_budget-{train_hard_budget_for_eval(wc)}+"
            "train_soft_budget_param-null/"
            "method.bundle"
        ),

    output:
        eval_data=
            "extra/output/eval_results/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-null/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
        eval_time=
            "extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-null/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        classifier_bundle_path=lambda wildcards: (
            f"extra/output/classifiers/masked_mlp_classifier/"
            f"dataset-{wildcards.dataset}+"
            f"instance_idx-{wildcards.dataset_instance_idx}/"
            f"seed-{wildcards.dataset_instance_idx}.bundle"
        ),
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output.eval_data} \
            classifier_bundle_path={params.classifier_bundle_path} \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget={wildcards.eval_hard_budget}

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output.eval_time}'
        """

def soft_trained_bundle(wc):
    maxb = MAX_HARD_BUDGET_PER_METHOD_DATASET[wc.method][wc.dataset]
    if int(wc.train_hard_budget) != int(maxb):
        raise ValueError(
            f"Soft-budget eval requested for non-max hard budget "
            f"{wc.train_hard_budget} (max is {maxb})"
        )
    return (
        f"extra/output/trained_methods/{wc.method}/"
        f"dataset-{wc.dataset}+"
        f"instance_idx-{wc.dataset_instance_idx}/"
        f"{wc.pretrain_folder}"
        f"train_seed-{wc.train_seed}+"
        f"train_hard_budget-{maxb}+"
        f"train_soft_budget_param-{wc.train_soft_budget_param}/"
        "method.bundle"
    )

rule eval_method_soft:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",
        soft_trained_bundle,
    output:
        eval_data="extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-null/"
                            "eval_data.csv",
        eval_time="extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-null/"
                            "eval_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        classifier_bundle_path=lambda wildcards: (
            f"extra/output/classifiers/masked_mlp_classifier/"
            f"dataset-{wildcards.dataset}+"
            f"instance_idx-{wildcards.dataset_instance_idx}/"
            f"seed-{wildcards.dataset_instance_idx}.bundle"
        ),
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output.eval_data} \
            classifier_bundle_path={params.classifier_bundle_path} \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget=null

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output.eval_time}'
        """

rule add_eval_metadata_to_eval_data:
    input:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col eval_soft_budget_param=""
        """

# Convert the `prev_selections_performed` (list[int]) and `selection_performed` columns into `selections_performed` (int)
rule count_selections:
    input:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py count_selections {input} {output}
        """

# Add some metadata columns from training
rule add_train_metadata_to_eval_data:
    input:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col afa_method={wildcards.method} \
            --col dataset={wildcards.dataset} \
            --col train_seed={wildcards.train_seed} \
            --col train_hard_budget={wildcards.train_hard_budget} \
            --col train_soft_budget_param={wildcards.train_soft_budget_param}
        """


# Make sure that only one of train_soft_budget_param and eval_soft_budget_param is set, if any
rule validate_hard_budget_and_soft_budget_param:
    input:
        "extra/output/eval_results4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results5/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py validate_budgets {input} {output}
        """

# Instead of two separate classifier columns, the plotting script expects tidy data
rule pivot_long_classifier:
    input:
        "extra/output/eval_results5/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results6/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py pivot_long_classifier {input} {output}
        """

rule merge_eval:
    input:
        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_budget}+"
                            f"train_soft_budget_param-null/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_budget, eval_budget) in HARD_BUDGET_EVAL_PAIRS_PER_METHOD[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_budget}+"
                            f"train_soft_budget_param-null/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_budget, eval_budget) in HARD_BUDGET_EVAL_PAIRS_PER_METHOD[method][dataset]
        ] +

        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{MAX_HARD_BUDGET_PER_METHOD_DATASET[method][dataset]}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-null/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for train_soft_budget_param in SOFT_BUDGET_PARAMS_PER_METHOD[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{MAX_HARD_BUDGET_PER_METHOD_DATASET[method][dataset]}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-null/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for train_soft_budget_param in SOFT_BUDGET_PARAMS_PER_METHOD[method][dataset]
        ]
    output:
        "extra/output/merged_eval_results/Greedy_and_static.csv",
    shell:
        "csvstack {input} > {output}"

rule plot:
    input:
        "extra/output/merged_eval_results/Greedy_and_static.csv",
    output:
        directory("extra/output/plot_results/Greedy_and_static"),
    shell:
        """
        python scripts/plotting/plot_eval.py {input} {output}
        """
