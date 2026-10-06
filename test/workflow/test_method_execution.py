"""Method routing through the ordinary Snakemake command boundary."""

from pathlib import Path

import pytest

from test.workflow.submission_harness import WorkflowHarness


def test_training_and_evaluation_have_independent_method_devices(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {
        "defaults": {"training": "cpu", "evaluation": "cuda"},
        "methods": {"alpha": {"training": "cuda", "evaluation": "cpu"}},
    }

    result = workflow.run("--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout + result.stderr
    assert commands.count("rule train_method:") == 2
    assert commands.count("rule eval_method:") == 2
    assert "device=cuda" in commands
    assert "device=cpu" in commands
    alpha_training = commands.split("python scripts/train_method/alpha.py", 1)[
        1
    ].split("END_TIME", 1)[0]
    beta_training = commands.split("python scripts/train_method/beta.py", 1)[
        1
    ].split("END_TIME", 1)[0]
    assert "device=cuda" in alpha_training
    assert "device=cpu" in beta_training


@pytest.mark.pipeline
@pytest.mark.parametrize("profile", ["mixed-gres", "mixed-gpus"])
def test_mixed_submissions_map_to_site_allocations(
    tmp_path: Path, profile: str
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {
        "defaults": {"training": "cpu", "evaluation": "cuda"},
        "methods": {"alpha": {"training": "cuda", "evaluation": "cpu"}},
    }

    result = workflow.run(
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles" / profile),
        "--slurm-init-seconds-before-status-checks",
        "0",
        "--seconds-between-status-checks",
        "1",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    submissions = workflow.submissions()
    assert len(submissions) == 4
    for args in submissions:
        comment = args[args.index("--comment") + 1]
        gpu = ("train_method" in comment and "alpha" in comment) or (
            "eval_method" in comment and "beta" in comment
        )
        assert args[args.index("-p") + 1] == (
            ("gpu-queue" if gpu else "cpu-queue")
            if profile == "mixed-gres"
            else ("accelerated" if gpu else "standard")
        )
        assert args[args.index("-A") + 1] == (
            ("gpu-account" if gpu else "cpu-account")
            if profile == "mixed-gres"
            else ("other-gpu" if gpu else "other-cpu")
        )
        request = (
            "--gres=gpu:T4:1" if profile == "mixed-gres" else "--gpus=a100:1"
        )
        assert (request in args) is gpu
        assert (
            not any(arg.startswith(("--gpus", "--gres")) for arg in args)
            if not gpu
            else True
        )
        assert args[args.index("-t") + 1] == "600"
        assert args[args.index("--mem") + 1] == "4000"
        assert "--cpus-per-task=8" in args
    calls = workflow.script_arguments()
    assert len(calls) == 4
    for script, args in calls:
        method = (
            "alpha"
            if "alpha" in script
            or "/alpha/" in args.get("method_bundle_path", "")
            else "beta"
        )
        expected = (
            "cuda"
            if ("train_method" in script) == (method == "alpha")
            else "cpu"
        )
        assert args["device"] == expected


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ({"execution": {"defaults": {"training": "tpu"}}}, "training/alpha"),
        ({"execution": {}, "device": "cuda"}, "global device"),
        ({"execution": {"defaults": {"trainng": "cuda"}}}, "trainng"),
        (
            {
                "execution": {"defaults": {"training": "cpu"}},
                "execution_site": {"cpu": {"gpu": 1}},
            },
            "CPU allocation",
        ),
        (
            {
                "execution": {"defaults": {"training": "cuda"}},
                "execution_site": {"gpu": {"gpu": 1, "gres": "gpu:T4:1"}},
            },
            "GPU allocation",
        ),
        (
            {
                "execution": {"defaults": {"training": "cuda"}},
                "execution_site": {"gpu": {"slurm_partition": "gpu-queue"}},
            },
            "GPU allocation",
        ),
        (
            {
                "execution": {
                    "defaults": {"training": "cuda", "slurm_partition": "site"}
                }
            },
            "slurm_partition",
        ),
    ],
)
def test_invalid_execution_fails_before_submission(
    tmp_path: Path,
    change: dict[str, object],
    diagnostic: str,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config.update(change)

    result = workflow.run("--executor", "slurm")

    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert workflow.submissions() == []


def test_method_arguments_cannot_override_the_resolved_device(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["method_options"] = {
        "alpha": {
            "train_script_name": "alpha",
            "eval_batch_size": 1,
            "method_specific_params": ["device=cuda"],
        },
        "beta": {"train_script_name": "beta", "eval_batch_size": 1},
    }

    result = workflow.run("--dry-run")

    assert result.returncode != 0
    assert "method_specific_params" in result.stdout + result.stderr
    assert "device" in result.stdout + result.stderr


def test_legacy_device_retains_site_defaults_and_local_cpu_execution(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["device"] = "cpu"

    result = workflow.run(
        "--executor",
        "local",
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles/vera"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout + result.stderr
    assert "deprecated" in commands
    assert "slurm_partition=vera" in commands
    assert "slurm_account=C3SE2026-1-12" in commands
    assert "runtime=6000" in commands
    assert "cpus_per_task=8" in commands


def test_rule_resource_override_cannot_contradict_cpu_device(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)

    result = workflow.run("--dry-run", "--set-resources", "train_method:gpu=1")

    assert result.returncode != 0
    assert "Conflicting allocation" in result.stdout + result.stderr
    assert workflow.submissions() == []


def test_downstream_conflict_is_detected_before_any_submission(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution_site"] = {
        "cpu": {
            "slurm_partition": "cpu-queue",
            "slurm_account": "cpu-account",
        },
    }

    result = workflow.run(
        "--executor", "slurm", "--set-resources", "eval_method:gpu=1"
    )

    assert result.returncode != 0
    assert "Conflicting allocation" in result.stdout + result.stderr
    assert workflow.submissions() == []


def test_unconverted_pretraining_preserves_legacy_device(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config.update(
        {
            "device": "cuda",
            "methods": ["alpha"],
            "method_options": {
                "alpha": {
                    "train_script_name": "alpha",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                },
            },
            "pretrain_mapping": {"shared": {"pretrain_script_name": "shared"}},
        }
    )

    result = workflow.run("--dry-run", target="all_pretrain_models")

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout + result.stderr
    pretraining = commands.split("python scripts/pretrain_model/shared.py", 1)[
        1
    ].split("END_TIME", 1)[0]
    assert "device=cuda" in pretraining
    assert "pretrain_seed-0/model.bundle" in pretraining


def test_evaluation_only_validates_the_selected_stage(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    trained = workflow.run("--executor", "local")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    workflow.config["execution"] = {
        "defaults": {"training": "cuda", "evaluation": "cpu"},
    }
    workflow.config["execution_site"] = {
        "cpu": {
            "slurm_partition": "cpu-queue",
            "slurm_account": "cpu-account",
        },
    }

    result = workflow.run(
        "--dry-run",
        "--forcerun",
        "eval_method",
        "--snakefile",
        "extra/workflow/snakefiles/orchestration/pipeline_no_train.smk",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "device=cpu" in result.stdout + result.stderr
    assert "rule train_method:" not in result.stdout + result.stderr
