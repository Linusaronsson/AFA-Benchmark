"""Requested walltime through real Snakemake planning and job resources."""

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from test.workflow.submission_harness import WorkflowHarness
from test.workflow.test_full_reproduction import planned_jobs


def job_runtimes(output: str, rule: str) -> dict[str, int]:
    blocks = re.findall(rf"rule {rule}:\n(.*?)(?=\n\n)", output, re.DOTALL)
    return {
        re.search(r"method=(\w+)", block)[1]: int(
            re.search(r"runtime=(\d+)", block)[1]
        )
        for block in blocks
    }


def test_site_walltime_overrides_arrhenius_legacy_rule_limit(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    # Exercise the real Arrhenius profile without requiring its images.
    site = tmp_path / "workflow/profiles/site/arrhenius/site.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "walltime": {
                    "default_hours": 2,
                    "rules": {
                        "train_method": {
                            "default_hours": 8,
                            "methods": {
                                "alpha": {
                                    "default_hours": 4,
                                    "datasets": {"cube": 1.01},
                                }
                            },
                        }
                    },
                }
            }
        )
    )

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(site.parent),
        target="all_train_methods",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert job_runtimes(result.stdout + result.stderr, "train_method") == {
        "alpha": 61,
        "beta": 480,
    }


@pytest.mark.parametrize(
    "profile", ["vera", "alvis", "examples/mixed-gres", "examples/mixed-gpus"]
)
def test_other_site_profiles_do_not_shadow_the_walltime_table(
    tmp_path: Path, profile: str
) -> None:
    workflow = WorkflowHarness(tmp_path)
    site = tmp_path / "workflow/profiles/site" / profile / "site.yaml"
    config = yaml.safe_load(site.read_text())
    config["walltime"] = {"rules": {"train_method": {"default_hours": 3}}}
    site.write_text(yaml.safe_dump(config))

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(site.parent),
        target="all_train_methods",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert job_runtimes(result.stdout + result.stderr, "train_method") == {
        "alpha": 180,
        "beta": 180,
    }


def test_planning_and_job_records_use_the_resolved_walltime(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["walltime"] = {
        "default_hours": 2,
        "rules": {
            "train_method": {
                "methods": {"alpha": {"datasets": {"cube": 1.01}}}
            },
            "eval_method": {"default_hours": 0.5},
        },
    }

    plan = workflow.plan("--executor", "local")
    assert plan.returncode == 0, plan.stdout + plan.stderr
    planned = {
        (job["rule"], json.loads(job["wildcards"])["method"]): int(
            job["time_limit_minutes"]
        )
        for job in workflow.planned_jobs()
    }
    assert planned == {
        ("train_method", "alpha"): 61,
        ("train_method", "beta"): 120,
        ("eval_method", "alpha"): 30,
        ("eval_method", "beta"): 30,
    }

    run = workflow.run("--executor", "local")
    assert run.returncode == 0, run.stdout + run.stderr
    records = [
        json.loads(path.read_text())
        for path in workflow.output.rglob("*.job_record.json")
    ]
    assert {
        (record["stage"], record["name"]): record["time_limit_minutes"]
        for record in records
    } == {
        ("training", "alpha"): 61,
        ("training", "beta"): 120,
        ("evaluation", "alpha"): 30,
        ("evaluation", "beta"): 30,
    }


def test_pretraining_and_classifier_limits_across_datasets_and_realizations(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    shutil.rmtree(workflow.output / "trained_classifiers")
    workflow.config.update(
        {
            "datasets": ["cube", "cube_nm"],
            "dataset_realization_indices": [0, 1],
            "method_options": {
                "alpha": {
                    "train_script_name": "alpha",
                    "pretrained_model_name": "shared",
                },
                "beta": {
                    "train_script_name": "beta",
                    "pretrained_model_name": "other",
                },
            },
            "pretrain_mapping": {
                "shared": {"pretrain_script_name": "shared"},
                "other": {"pretrain_script_name": "other"},
            },
            "walltime": {
                "default_hours": 2,
                "rules": {
                    "train_classifier": {"default_hours": 0.5},
                    "pretrain_model": {
                        "default_hours": 8,
                        "pretrained_models": {
                            "shared": {
                                "default_hours": 3,
                                "datasets": {"cube": 0.05},
                            },
                            "other": {"datasets": {"cube_nm": 0.01}},
                        },
                    },
                },
            },
        }
    )

    result = workflow.run("--dry-run", target="all_pretrain_models")
    assert result.returncode == 0, result.stdout + result.stderr
    jobs = planned_jobs(result.stdout + result.stderr)
    expected = {
        ("shared", "cube"): 3,
        ("shared", "cube_nm"): 180,
        ("other", "cube"): 480,
        ("other", "cube_nm"): 1,
    }
    found = {}
    classifiers = []
    for job in jobs:
        if job["rule"] == "pretrain_model":
            wc = dict(
                item.split("=", 1) for item in job["wildcards"].split(", ")
            )
            minutes = int(re.search(r"runtime=(\d+)", job["resources"])[1])
            key = (wc["pretrained_model_name"], wc["dataset"])
            assert minutes == expected[key]
            found.setdefault(key, set()).add(wc["dataset_realization_index"])
        elif job["rule"] == "train_classifier":
            classifiers.append(job)
            assert "runtime=30" in job["resources"]
    assert found == {key: {"0", "1"} for key in expected}
    assert len(classifiers) == 4


@pytest.mark.parametrize(
    "table",
    [
        {},
        {
            "rules": {
                "train_method": {"methods": {"alpha": {"default_hours": 1}}}
            }
        },
    ],
)
def test_sparse_or_absent_table_preserves_default_resource_expressions(
    tmp_path: Path, table: dict[str, object]
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["walltime"] = table
    result = workflow.plan(
        "--executor",
        "local",
        "--default-resources",
        "runtime=30+attempt",
        target="all_train_methods",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {
        json.loads(job["wildcards"])["method"]: int(job["time_limit_minutes"])
        for job in workflow.planned_jobs()
    } == {"alpha": 60 if table else 31, "beta": 31}


def test_explicit_runtime_override_wins_over_the_site_table(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/examples/mixed-gres"),
        "--set-resources",
        "train_method:runtime=17",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert job_runtimes(output, "train_method") == {"alpha": 17, "beta": 17}
    # --set-resources replaces the profile section, but not the site's table.
    assert job_runtimes(output, "eval_method") == {"alpha": 600, "beta": 600}


def test_invalid_walltime_fails_before_any_submission(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["walltime"] = {
        "rules": {"train_methd": {"default_hours": 4}}
    }
    result = workflow.run("--executor", "slurm")
    assert result.returncode != 0
    assert "train_methd" in result.stdout + result.stderr
    assert workflow.submissions() == []
