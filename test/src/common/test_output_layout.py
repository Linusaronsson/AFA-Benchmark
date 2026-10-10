"""
The native output layout, pinned to the paths the pipeline writes today.

Benchmark releases are restored at these paths and reference methods'
tables are found by them, so every expected value is a literal.
"""

import re

import pytest

from afabench.core.output_layout import (
    PRETRAIN_FOLDER_PATTERN,
    EvaluationRun,
    OutputLayout,
    TrainingRun,
    pretrain_seed_folder,
    pretrain_seed_in_folder,
)

LAYOUT = OutputLayout(root="output", initializer="cold", eval_split="test")


def test_dataset_bundle() -> None:
    assert (
        LAYOUT.dataset_bundle(
            dataset="cube", dataset_realization_index=1, split="val"
        )
        == "output/datasets/cube/1/val.bundle"
    )


def test_dataset_folder_holds_every_realization() -> None:
    assert LAYOUT.dataset_folder(dataset="cube") == "output/datasets/cube"


def test_dataset_generation_job_record() -> None:
    assert (
        LAYOUT.dataset_generation_job_record(
            dataset="cube", dataset_realization_index=0
        )
        == "output/datasets/cube/0/dataset_generation.job_record.json"
    )


def test_initializer_tag() -> None:
    assert LAYOUT.initializer_tag == "initializer-cold"


def test_external_classifier_bundle() -> None:
    assert (
        LAYOUT.classifier_bundle(
            dataset="cube", dataset_realization_index=1, method=None
        )
        == "output/trained_classifiers/initializer-cold/"
        "dataset-cube+realization_index-1.bundle"
    )


def test_built_in_classifier_bundle() -> None:
    assert (
        LAYOUT.classifier_bundle(
            dataset="cube", dataset_realization_index=1, method="beta"
        )
        == "output/trained_classifiers/initializer-cold/"
        "method-beta+dataset-cube+realization_index-1.bundle"
    )


def test_classifier_job_record_sits_beside_its_bundle() -> None:
    assert (
        LAYOUT.classifier_job_record(
            dataset="cube", dataset_realization_index=1, method="beta"
        )
        == "output/trained_classifiers/initializer-cold/"
        "method-beta+dataset-cube+realization_index-1.job_record.json"
    )


PRETRAINED_MODEL_FOLDER = (
    "output/pretrained_models/initializer-cold/shared/"
    "dataset-cube+realization_index-1/pretrain_seed-1/"
)


def test_pretrained_model_bundle() -> None:
    assert (
        LAYOUT.pretrained_model_bundle(
            pretrained_model_name="shared",
            dataset="cube",
            dataset_realization_index=1,
            pretrain_seed=1,
        )
        == f"{PRETRAINED_MODEL_FOLDER}model.bundle"
    )


def test_pretraining_job_record() -> None:
    assert (
        LAYOUT.pretraining_job_record(
            pretrained_model_name="shared",
            dataset="cube",
            dataset_realization_index=1,
            pretrain_seed=1,
        )
        == f"{PRETRAINED_MODEL_FOLDER}model.job_record.json"
    )


# beta has a pretraining stage and a hard budget, alpha neither: it trains
# with a soft-budget parameter.
BETA_RUN = TrainingRun(
    method="beta",
    dataset="cube",
    dataset_realization_index=1,
    pretrain_folder=pretrain_seed_folder(1),
    train_seed=1,
    train_hard_budget=2,
    train_soft_budget_param="null",
)
BETA_RUN_FOLDER = (
    "beta/dataset-cube+realization_index-1/pretrain_seed-1/"
    "train_seed-1+train_hard_budget-2+train_soft_budget_param-null/"
)
ALPHA_RUN = TrainingRun(
    method="alpha",
    dataset="cube",
    dataset_realization_index=0,
    pretrain_folder=pretrain_seed_folder(None),
    train_seed=0,
    train_hard_budget="null",
    train_soft_budget_param=0.5,
)
ALPHA_RUN_FOLDER = (
    "alpha/dataset-cube+realization_index-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-null+train_soft_budget_param-0.5/"
)


def test_method_bundle_with_pretraining_stage() -> None:
    assert (
        LAYOUT.method_bundle(BETA_RUN)
        == "output/trained_methods/initializer-cold/"
        f"{BETA_RUN_FOLDER}method.bundle"
    )


def test_method_bundle_without_pretraining_stage() -> None:
    assert (
        LAYOUT.method_bundle(ALPHA_RUN)
        == "output/trained_methods/initializer-cold/"
        f"{ALPHA_RUN_FOLDER}method.bundle"
    )


def test_training_job_record() -> None:
    assert (
        LAYOUT.training_job_record(ALPHA_RUN)
        == "output/trained_methods/initializer-cold/"
        f"{ALPHA_RUN_FOLDER}method.job_record.json"
    )


# Hard-budget evaluation of beta, soft-budget evaluation of alpha.
BETA_EVALUATION = EvaluationRun(
    training=BETA_RUN,
    eval_seed=1,
    eval_hard_budget=2,
    eval_soft_budget_param="null",
)
BETA_EVALUATION_FOLDER = (
    f"eval_split-test/initializer-cold/{BETA_RUN_FOLDER}"
    "eval_seed-1+eval_hard_budget-2+eval_soft_budget_param-null/"
)
ALPHA_EVALUATION = EvaluationRun(
    training=ALPHA_RUN,
    eval_seed=0,
    eval_hard_budget="null",
    eval_soft_budget_param=0.2,
)
ALPHA_EVALUATION_FOLDER = (
    f"eval_split-test/initializer-cold/{ALPHA_RUN_FOLDER}"
    "eval_seed-0+eval_hard_budget-null+eval_soft_budget_param-0.2/"
)


def test_raw_evaluation_table_of_hard_budget_run() -> None:
    assert (
        LAYOUT.raw_evaluation_table(BETA_EVALUATION)
        == f"output/eval_results/{BETA_EVALUATION_FOLDER}"
        "eval_data.parquet"
    )


def test_raw_evaluation_table_of_soft_budget_run() -> None:
    assert (
        LAYOUT.raw_evaluation_table(ALPHA_EVALUATION)
        == f"output/eval_results/{ALPHA_EVALUATION_FOLDER}"
        "eval_data.parquet"
    )


def test_transformed_evaluation_table() -> None:
    assert (
        LAYOUT.transformed_evaluation_table(ALPHA_EVALUATION)
        == f"output/eval_results_transformed/{ALPHA_EVALUATION_FOLDER}"
        "eval_data.parquet"
    )


def test_evaluation_job_record_sits_beside_its_table() -> None:
    assert (
        LAYOUT.evaluation_job_record(BETA_EVALUATION)
        == f"output/eval_results/{BETA_EVALUATION_FOLDER}"
        "eval_data.job_record.json"
    )


def test_transformation_job_record_sits_beside_its_table() -> None:
    assert (
        LAYOUT.transformation_job_record(BETA_EVALUATION)
        == f"output/eval_results_transformed/{BETA_EVALUATION_FOLDER}"
        "eval_data.job_record.json"
    )


def test_wildcards_give_a_rule_pattern() -> None:
    assert LAYOUT.raw_evaluation_table(EvaluationRun.wildcards()) == (
        "output/eval_results/eval_split-test/initializer-cold/"
        "{method}/dataset-{dataset}+"
        "realization_index-{dataset_realization_index}/{pretrain_folder}/"
        "train_seed-{train_seed}+train_hard_budget-{train_hard_budget}+"
        "train_soft_budget_param-{train_soft_budget_param}/"
        "eval_seed-{eval_seed}+eval_hard_budget-{eval_hard_budget}+"
        "eval_soft_budget_param-{eval_soft_budget_param}/eval_data.parquet"
    )


def test_wildcards_of_one_pretrain_folder_kind() -> None:
    run = TrainingRun.wildcards(pretrain_folder=pretrain_seed_folder(None))

    assert LAYOUT.training_job_record(run) == (
        "output/trained_methods/initializer-cold/"
        "{method}/dataset-{dataset}+"
        "realization_index-{dataset_realization_index}/NO_PRETRAIN/"
        "train_seed-{train_seed}+train_hard_budget-{train_hard_budget}+"
        "train_soft_budget_param-{train_soft_budget_param}/"
        "method.job_record.json"
    )


@pytest.mark.parametrize(
    ("folder", "pretrain_seed"),
    [("pretrain_seed-3", "3"), ("NO_PRETRAIN", None)],
)
def test_pretrain_folder_wildcard_reads_back_its_seed(
    folder: str, pretrain_seed: str | None
) -> None:
    assert re.fullmatch(PRETRAIN_FOLDER_PATTERN, folder)
    assert pretrain_seed_in_folder(folder) == pretrain_seed


def test_pretrain_folder_wildcard_rejects_other_folders() -> None:
    assert not re.fullmatch(PRETRAIN_FOLDER_PATTERN, "pretrain_seed-x")
    with pytest.raises(ValueError, match="'pretrain_seed-x'"):
        pretrain_seed_in_folder("pretrain_seed-x")
