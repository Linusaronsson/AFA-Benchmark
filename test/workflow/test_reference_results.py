"""
Compare local methods with restored reference results (#41).

`reference_methods` names methods whose plotting-ready tables come from a
restored benchmark release. Through the ordinary Snakemake command boundary,
with stubbed scripts, these tests check what the workflow schedules: only the
local methods' work, never the production of a reference table, and no
reference table of another evaluation split, initializer or budget.
"""

import os
import re
from pathlib import Path

import pytest

from test.workflow.submission_harness import WorkflowHarness

TAG = "initializer-cold"
ALPHA_HARD_BUDGET_TABLE = (
    f"extra/output/smoke/eval_results_transformed/eval_split-test/{TAG}/alpha/"
    "dataset-cube+realization_index-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-1+train_soft_budget_param-null/"
    "eval_seed-0+eval_hard_budget-1+eval_soft_budget_param-null/"
    "eval_data.parquet"
)
ALPHA_SOFT_BUDGET_TABLE = (
    f"extra/output/smoke/eval_results_transformed/eval_split-test/{TAG}/alpha/"
    "dataset-cube+realization_index-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-null+train_soft_budget_param-0.5/"
    "eval_seed-0+eval_hard_budget-null+eval_soft_budget_param-null/"
    "eval_data.parquet"
)
# Older than every prerequisite the harness creates, as a download of an
# old release can be: the workflow must not take that as outdated.
RELEASE_MTIME = 1_600_000_000


def adopter_workflow(root: Path, *tables: str) -> WorkflowHarness:
    """Beta is the adopter's method; alpha's tables are restored."""
    workflow = WorkflowHarness(root)
    workflow.config["methods"] = ["beta"]
    workflow.config["reference_methods"] = ["alpha"]
    workflow.config["method_sets"] = {
        "comparison": ["alpha", "beta"],
        "published": ["alpha"],
    }
    for table in tables:
        path = root / table
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("restored")
        os.utime(path, (RELEASE_MTIME, RELEASE_MTIME))
    return workflow


def job_counts(output: str) -> dict[str, int]:
    stats = output.split("Job stats:", 1)[1].split("\n\n\n", 1)[0]
    return {
        rule: int(count)
        for rule, count in re.findall(
            r"^(\w+)\s+(\d+)$", stats, flags=re.MULTILINE
        )
    }


def planned_outputs(output: str) -> dict[str, list[str]]:
    """Output files of every planned job, by rule."""
    outputs: dict[str, list[str]] = {}
    for rule, files in re.findall(
        r"^rule (\w+):\n(?:    .*\n)*?    output: (.*)$",
        output,
        flags=re.MULTILINE,
    ):
        outputs.setdefault(rule, []).extend(files.split(", "))
    return outputs


def test_only_local_methods_are_produced_for_a_comparison_with_references(
    tmp_path: Path,
) -> None:
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)

    plan = workflow.run("--dry-run", target="all")

    output = plan.stdout + plan.stderr
    assert plan.returncode == 0, output
    assert job_counts(output) == {
        "all": 1,
        "eval_method": 1,
        "merge_eval_perf": 1,
        "collect_job_records": 1,
        "plot_eval_perf": 2,
        "plot_time": 1,
        "split_by_classifier_type": 1,
        "train_method": 1,
        "transform_eval_data": 1,
        "total": 10,
    }, output
    merge = output.split("rule merge_eval_perf:", 1)[1].split("output:", 1)[0]
    assert ALPHA_HARD_BUDGET_TABLE in merge
    assert "/beta/" in merge
    produced = [
        path for paths in planned_outputs(output).values() for path in paths
    ]
    assert [path for path in produced if "/beta/" in path], output
    assert not [path for path in produced if "/alpha/" in path], output
    assert not [path for path in produced if "published" in path], output


def test_missing_reference_table_fails_the_plan_instead_of_being_produced(
    tmp_path: Path,
) -> None:
    # The release has alpha's hard-budget table only; the soft-budget one
    # must not be filled in by a hard-budget table or by evaluating alpha.
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)
    workflow.config["soft_budget_params"]["alpha"] = {"default": [[0.5, None]]}

    plan = workflow.run("--dry-run", target="all")

    output = plan.stdout + plan.stderr
    assert plan.returncode != 0, output
    assert "MissingInputException" in output, output
    assert ALPHA_SOFT_BUDGET_TABLE in output, output
    assert "Job stats:" not in output, output


@pytest.mark.parametrize(
    ("setting", "expected_folder"),
    [
        ({"eval_dataset_split": "val"}, f"eval_split-val/{TAG}/alpha/"),
        (
            {"initializer": "missingness"},
            "eval_split-test/initializer-missingness/alpha/",
        ),
    ],
)
def test_reference_table_of_another_split_or_initializer_is_not_compared(
    tmp_path: Path, setting: dict[str, str], expected_folder: str
) -> None:
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)
    workflow.config |= setting

    plan = workflow.run("--dry-run", target="all")

    output = plan.stdout + plan.stderr
    assert plan.returncode != 0, output
    assert "MissingInputException" in output, output
    assert expected_folder in output, output
    assert ALPHA_HARD_BUDGET_TABLE not in output, output


def test_reference_method_that_is_also_produced_is_rejected(
    tmp_path: Path,
) -> None:
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)
    workflow.config["methods"] = ["alpha", "beta"]

    plan = workflow.run("--dry-run", target="all")

    output = plan.stdout + plan.stderr
    assert plan.returncode != 0, output
    assert "['alpha'] are both in methods and in reference_methods" in output


def test_reference_method_without_method_options_is_rejected(
    tmp_path: Path,
) -> None:
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)
    workflow.config["reference_methods"] = ["alpha", "gamma"]

    plan = workflow.run("--dry-run", target="all")

    output = plan.stdout + plan.stderr
    assert plan.returncode != 0, output
    assert "Reference methods ['gamma'] are not in method_options" in output
