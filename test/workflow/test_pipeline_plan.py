"""
The shared planner reads a real workflow's graph of run and reused jobs (#159).

The statuses and reasons are Snakemake's own decisions. The plans come from the interactive preview's planning child,
`python -m afabench.pipeline_plan`, over the harness's workflow. Fresh
work, up-to-date, missing and stale outputs, dependency-driven reruns,
target scoping, a stage that is off, and budget settings are covered;
`test_compute_estimate_planning.py` covers allocations and invalid
invocations.
"""

import os
import shutil
import time
from pathlib import Path

from afabench.pipeline_plan.planning import PipelinePlan, PlannedJob
from afabench.pipeline_preview.model import (
    SHARED_PREREQUISITES,
    consuming_methods,
    existing_inputs,
    group_of,
    job_details,
    preview_tree,
)
from test.workflow.submission_harness import WorkflowHarness

type Key = tuple[str, str | None]


def by_key(plan: PipelinePlan) -> dict[Key, PlannedJob]:
    """
    Index the jobs by rule and method, one of each in the harness.

    The plots of a method set are told apart by classifier type instead.
    """
    jobs = {
        (
            job.rule,
            job.wildcards.get("method")
            or job.wildcards.get("classifier_type"),
        ): job
        for job in plan.jobs
    }
    assert len(jobs) == len(plan.jobs)
    return jobs


def statuses(plan: PipelinePlan) -> dict[Key, str]:
    return {key: job.status for key, job in by_key(plan).items()}


def reasons(job: PlannedJob) -> list[str]:
    return [reason.kind for reason in job.reasons]


def needs(plan: PipelinePlan, job: PlannedJob) -> set[Key]:
    jobs = by_key(plan)
    return {
        key for key, needed in jobs.items() if needed.jobid in job.dependencies
    }


def test_fresh_work_is_planned_over_its_reusable_prerequisites(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)

    # The harness's dataset and classifier bundles exist; nothing else does.
    plan = workflow.planned_graph("--executor", "local", target="all")

    jobs = by_key(plan)
    assert statuses(plan) == {
        ("dataset_generation", None): "reuse",
        ("train_classifier", None): "reuse",
        ("train_method", "alpha"): "run",
        ("train_method", "beta"): "run",
        ("eval_method", "alpha"): "run",
        ("eval_method", "beta"): "run",
        ("transform_eval_data", "alpha"): "run",
        ("transform_eval_data", "beta"): "run",
        ("merge_eval_perf", None): "run",
        ("split_by_classifier_type", None): "run",
        ("collect_job_records", None): "run",
        ("plot_eval_perf", "builtin"): "run",
        ("plot_eval_perf", "external"): "run",
        ("plot_time", None): "run",
    }
    # Each shared prerequisite is one job, needed by both methods.
    assert needs(plan, jobs[("train_method", "alpha")]) == {
        ("dataset_generation", None),
        ("train_classifier", None),
    }
    assert needs(plan, jobs[("eval_method", "beta")]) == {
        ("dataset_generation", None),
        ("train_classifier", None),
        ("train_method", "beta"),
    }
    assert consuming_methods(jobs[("train_classifier", None)], plan) == [
        "alpha",
        "beta",
    ]
    assert reasons(jobs[("train_method", "alpha")]) == ["missing_output"]
    assert reasons(jobs[("eval_method", "alpha")]) == [
        "missing_output",
        "input_from_rerun",
    ]
    assert reasons(jobs[("dataset_generation", None)]) == [
        "outputs_up_to_date"
    ]
    # A job without a job record still has its paths and dependencies.
    merge = jobs[("merge_eval_perf", None)]
    assert merge.identity is None
    assert needs(plan, merge) == {
        ("transform_eval_data", "alpha"),
        ("transform_eval_data", "beta"),
    }
    assert len(merge.inputs) == 2
    # The classifier's name is not a method's.
    classifier = jobs[("train_classifier", None)]
    assert classifier.identity is not None
    assert classifier.identity.name == "masked_mlp_classifier"
    assert workflow.submissions() == []


def test_complete_outputs_are_reused_and_missing_or_stale_ones_rerun(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    ran = workflow.run("--executor", "local")
    assert ran.returncode == 0, ran.stdout + ran.stderr

    complete = workflow.planned_graph("--executor", "local")

    # A valid plan with nothing to run still holds the reused jobs.
    assert set(statuses(complete).values()) == {"reuse"}
    assert complete.scheduled_jobs == []
    assert len(complete.jobs) == 6

    jobs = by_key(complete)
    alpha_bundle = jobs[("train_method", "alpha")].outputs[0]
    shutil.rmtree(tmp_path / alpha_bundle)

    # Status follows Snakemake's decision, not file existence: nothing
    # needs the missing bundle while alpha's evaluation of it exists.
    assert set(
        statuses(workflow.planned_graph("--executor", "local")).values()
    ) == {"reuse"}

    (tmp_path / jobs[("eval_method", "alpha")].outputs[0]).unlink()
    beta_bundle = tmp_path / jobs[("train_method", "beta")].outputs[0]
    # Newer than beta's evaluation of it
    newer = time.time() + 600
    for path in [beta_bundle, *beta_bundle.iterdir()]:
        os.utime(path, (newer, newer))

    changed = workflow.planned_graph("--executor", "local")

    assert statuses(changed) == {
        ("dataset_generation", None): "reuse",
        ("train_classifier", None): "reuse",
        ("train_method", "alpha"): "run",
        ("train_method", "beta"): "reuse",
        ("eval_method", "alpha"): "run",
        ("eval_method", "beta"): "run",
    }
    jobs = by_key(changed)
    assert reasons(jobs[("train_method", "alpha")]) == ["missing_output"]
    assert reasons(jobs[("eval_method", "alpha")]) == [
        "missing_output",
        "input_from_rerun",
    ]
    assert jobs[("eval_method", "alpha")].reasons[1].files == (alpha_bundle,)
    assert reasons(jobs[("eval_method", "beta")]) == ["updated_input"]
    assert jobs[("eval_method", "beta")].reasons[0].files == (
        str(beta_bundle.relative_to(tmp_path)),
    )


def test_the_plan_is_scoped_to_the_requested_targets(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)

    plan = workflow.planned_graph(
        "--executor", "local", target="all_train_methods"
    )

    assert statuses(plan) == {
        ("dataset_generation", None): "reuse",
        ("train_classifier", None): "reuse",
        ("train_method", "alpha"): "run",
        ("train_method", "beta"): "run",
    }


def test_a_stage_that_is_off_has_no_jobs_and_its_outputs_are_existing_inputs(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["stages"] = {"dataset_generation": False}

    plan = workflow.planned_graph("--executor", "local")

    jobs = by_key(plan)
    assert ("dataset_generation", None) not in jobs
    classifier = jobs[("train_classifier", None)]
    assert needs(plan, classifier) == set()
    assert existing_inputs(classifier, plan) == list(classifier.inputs)
    assert all(
        path.startswith("output/smoke/datasets/cube/0/")
        for path in classifier.inputs
    )
    # A job of the plan makes the method bundle, so it is no existing input.
    assert existing_inputs(jobs[("eval_method", "alpha")], plan) == [
        "output/smoke/datasets/cube/0/test.bundle"
    ]


def test_training_and_evaluation_budgets_are_planned_as_distinct_jobs(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["eval_hard_budgets"] = {"default": [1, 2]}
    options = workflow.config["method_options"]
    assert isinstance(options, dict)
    # Alpha trains once under hard budget 2 and evaluates at 1 and 2; beta
    # also runs the soft-budget setting on cube.
    options["alpha"]["train_hard_budget"] = {"cube": 2}
    options["beta"]["soft_budget_datasets"] = ["cube"]
    workflow.config["soft_budget_params"] = {
        "alpha": {"default": []},
        "beta": {"default": [[0.3, 0.5]]},
    }

    plan = workflow.planned_graph("--executor", "local")

    def budgets(stage: str) -> set[tuple[object, ...]]:
        return {
            (
                job.identity.name,
                job.identity.train_hard_budget,
                job.identity.train_soft_budget_param,
                job.identity.eval_hard_budget,
                job.identity.eval_soft_budget_param,
            )
            for job in plan.jobs
            if job.identity is not None and job.identity.stage == stage
        }

    assert budgets("training") == {
        ("alpha", 2, None, None, None),
        ("beta", 1, None, None, None),
        ("beta", 2, None, None, None),
        ("beta", None, 0.3, None, None),
    }
    assert budgets("evaluation") == {
        ("alpha", 2, None, 1, None),
        ("alpha", 2, None, 2, None),
        ("beta", 1, None, 1, None),
        ("beta", 2, None, 2, None),
        ("beta", None, 0.3, None, 0.5),
    }
    # Every job is in one place of the tree and counts once, the shared
    # prerequisites included.
    tree = preview_tree(plan)
    assert sum(group.job_count for group in tree) == len(plan.jobs) == 11
    assert [(group.label, group.counts_label) for group in tree] == [
        ("dataset generation", "1 Reuse"),
        ("classifier training", "1 Reuse"),
        ("training", "4 Run"),
        ("evaluation", "5 Run"),
    ]


def test_a_planned_job_has_its_identity_paths_placement_and_command(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}
    profile = str(tmp_path / "workflow/profiles/site/examples/mixed-gres")

    plan = workflow.planned_graph(
        "--workflow-profile", profile, target="all_train_methods"
    )

    jobs = by_key(plan)
    alpha = jobs[("train_method", "alpha")]
    assert alpha.identity is not None
    assert (
        alpha.identity.stage,
        alpha.identity.name,
        alpha.identity.dataset_key,
        alpha.identity.dataset_realization_index,
        alpha.identity.train_hard_budget,
    ) == ("training", "alpha", "cube", 0, 1)
    assert reasons(alpha) == ["missing_output"]
    assert set(alpha.reasons[0].files) <= set(alpha.outputs)
    assert alpha.inputs == (
        "output/smoke/datasets/cube/0/train.bundle",
        "output/smoke/datasets/cube/0/val.bundle",
        "output/smoke/trained_classifiers/initializer-cold/"
        "dataset-cube+realization_index-0.bundle",
    )
    # The site's GPU allocation and the profile's resources
    assert (alpha.device, alpha.cpus, alpha.gpus) == ("cuda", 8, 1)
    assert alpha.time_limit_minutes == 600
    assert alpha.placement == {
        "slurm_partition": "gpu-queue",
        "slurm_account": "gpu-account",
        "gres": "gpu:T4:1",
        "mem_mb": "4000",
    }
    beta = jobs[("train_method", "beta")]
    assert beta.placement["slurm_partition"] == "cpu-queue"
    # The command Snakemake would run, formatted while planning
    assert alpha.shell_command is not None
    assert "python scripts/train_method/alpha.py" in alpha.shell_command
    assert f"save_path={alpha.outputs[0]}" in alpha.shell_command
    assert "device=cuda" in alpha.shell_command
    assert workflow.submissions() == []
    assert not workflow.arguments.exists()


def test_a_pretrained_model_is_a_shared_prerequisite_not_a_method(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    options = workflow.config["method_options"]
    assert isinstance(options, dict)
    options["alpha"]["pretrained_model_name"] = "shared"
    workflow.config["pretrain_mapping"] = {
        "shared": {"pretrain_script_name": "shared"}
    }

    plan = workflow.planned_graph("--executor", "local")

    jobs = by_key(plan)
    pretraining = jobs[("pretrain_model", None)]
    assert pretraining.status == "run"
    assert pretraining.identity is not None
    assert pretraining.identity.stage == "pretraining"
    assert pretraining.identity.name == "shared"
    assert group_of(pretraining, "method") == SHARED_PREREQUISITES
    assert consuming_methods(pretraining, plan) == ["alpha"]
    details = dict(job_details(pretraining, plan))
    assert details["Method"] == "not applicable"
    assert details["Pretrained model"] == "shared"
    assert pretraining.jobid in jobs[("train_method", "alpha")].dependencies
    assert pretraining.jobid not in jobs[("train_method", "beta")].dependencies
