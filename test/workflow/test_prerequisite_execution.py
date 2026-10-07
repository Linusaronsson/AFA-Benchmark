"""Independent prerequisite routing at the real Snakemake invocation boundary."""

import shutil
from pathlib import Path

import pytest

from test.workflow.submission_harness import SITE, WorkflowHarness


def prerequisite_workflow(root: Path) -> WorkflowHarness:
    workflow = WorkflowHarness(root)
    shutil.rmtree(
        root
        / "extra/output/trained_classifiers/initializer-cold/dataset-cube.bundle"
    )
    workflow.config["method_options"] = {
        "alpha": {
            "train_script_name": "alpha",
            "eval_batch_size": 1,
            "classifier": {"script_name": "special"},
        },
        "beta": {"train_script_name": "beta", "eval_batch_size": 1},
    }
    for script in [
        "train_classifier/masked_mlp_classifier.py",
        "train_classifier/special.py",
        "pretrain_model/shared.py",
        "pretrain_model/other.py",
    ]:
        path = root / "scripts" / script
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "scripts/train_method/alpha.py", path)
    # Script-boundary stubs require their declared bundle inputs to exist,
    # so an incorrectly wired dependency cannot pass by merely writing outputs.
    for path in (root / "scripts").rglob("*.py"):
        path.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "inputs = dict(arg.split('=', 1) for arg in sys.argv[1:])\n"
            "for key, value in inputs.items():\n"
            "    if key.endswith(('_bundle_path', '_dataset_path')) and value != 'null':\n"
            "        assert Path(value).is_dir(), (key, value)\n"
            + path.read_text()
        )
    return workflow


def test_external_and_method_classifiers_are_independent_of_policy_training(
    tmp_path: Path,
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    workflow.config["execution"] = {
        "defaults": {"classifier_training": "cuda", "training": "cpu"},
        "methods": {
            "alpha": {"classifier_training": "cpu", "training": "cuda"}
        },
    }

    result = workflow.run("--dry-run")

    commands = result.stdout + result.stderr
    assert result.returncode == 0, commands
    external = commands.split(
        "python scripts/train_classifier/masked_mlp_classifier.py", 1
    )[1].split("experiment@_global_=cube", 1)[0]
    specific = commands.split("python scripts/train_classifier/special.py", 1)[
        1
    ].split("experiment@_global_=cube", 1)[0]
    assert "device=cuda" in external
    assert "device=cpu" in specific
    assert "dataset-cube.bundle" in external
    assert "method-alpha+dataset-cube.bundle" in specific
    assert "seed=0" in external
    assert "seed=0" in specific
    assert commands.count("rule train_classifier:") == 1
    assert commands.count("rule train_classifier_for_method:") == 1


def add_pretrained_models(workflow: WorkflowHarness) -> None:
    workflow.config.update(
        {
            "methods": ["alpha", "beta", "gamma"],
            "method_options": {
                "alpha": {
                    "train_script_name": "alpha",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                    "classifier": {"script_name": "special"},
                },
                "beta": {
                    "train_script_name": "beta",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                },
                "gamma": {
                    "train_script_name": "gamma",
                    "pretrained_model_name": "other",
                    "eval_batch_size": 1,
                },
            },
            "soft_budget_params": {
                name: {"default": []} for name in ["alpha", "beta", "gamma"]
            },
            "pretrain_mapping": {
                "shared": {"pretrain_script_name": "shared"},
                "other": {"pretrain_script_name": "other"},
            },
        }
    )
    shutil.copyfile(
        workflow.root / "scripts/train_method/alpha.py",
        workflow.root / "scripts/train_method/gamma.py",
    )


def test_named_pretraining_is_deduplicated_and_independent_of_method_order(
    tmp_path: Path,
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["execution"] = {
        "defaults": {"pretraining": "cpu", "training": "cpu"},
        "pretrained_models": {"shared": "cuda"},
        "methods": {
            "alpha": {"training": "cpu"},
            "beta": {"training": "cuda"},
        },
    }

    for methods in [["alpha", "beta", "gamma"], ["gamma", "beta", "alpha"]]:
        workflow.config["methods"] = methods
        result = workflow.run("--dry-run")
        commands = result.stdout + result.stderr
        assert result.returncode == 0, commands
        assert commands.count("rule pretrain_model:") == 2
        assert commands.count("rule train_method:") == 3
        shared = commands.split("python scripts/pretrain_model/shared.py", 1)[
            1
        ].split("END_TIME", 1)[0]
        other = commands.split("python scripts/pretrain_model/other.py", 1)[
            1
        ].split("END_TIME", 1)[0]
        assert "device=cuda" in shared
        assert "device=cpu" in other
        assert "seed=0" in shared
        assert "smoke_test=True" in shared
        assert "initializer=cold" in shared
        assert "unmasker=direct" in shared
        assert "classifier_bundle_path=" in shared
        assert (
            "shared/dataset-cube+realization_index-0/pretrain_seed-0/model.bundle"
            in shared
        )
        for method in ["alpha", "beta"]:
            training = commands.split(
                f"python scripts/train_method/{method}.py", 1
            )[1].split("END_TIME", 1)[0]
            assert (
                "shared/dataset-cube+realization_index-0/pretrain_seed-0/model.bundle"
                in training
            )
            assert "hard_budget=1" in training
            assert "soft_budget_param=null" in training


@pytest.mark.parametrize("prerequisite", ["external", "method", "pretraining"])
def test_prerequisite_script_arguments_cannot_override_device_before_submission(
    tmp_path: Path, prerequisite: str
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    if prerequisite == "external":
        workflow.config["classifier_names"] = {
            "default": {
                "script_name": "masked_mlp_classifier",
                "script_params": ["+device=cuda"],
            }
        }
    elif prerequisite == "method":
        workflow.config["method_options"] = {
            "alpha": {
                "train_script_name": "alpha",
                "pretrained_model_name": "shared",
                "eval_batch_size": 1,
                "classifier": {
                    "script_name": "special",
                    "script_params": ["device=cuda"],
                },
            },
            "beta": {"train_script_name": "beta", "eval_batch_size": 1},
        }
        workflow.config["methods"] = ["alpha", "beta"]
    else:
        workflow.config["pretrain_mapping"] = {
            "shared": {
                "pretrain_script_name": "shared",
                "pretrain_params": ["device=cuda"],
            },
            "other": {"pretrain_script_name": "other"},
        }
    workflow.config["execution_site"] = SITE

    result = workflow.run("--executor", "slurm")

    assert result.returncode != 0
    assert "cannot set device" in result.stdout + result.stderr
    assert workflow.submissions() == []


@pytest.mark.parametrize(
    ("execution", "diagnostic"),
    [
        ({"defaults": {"classifier_training": "tpu"}}, "classifier_training/"),
        ({"defaults": {"pretraining": "tpu"}}, "pretraining/"),
        ({"pretrained_models": {"shared": "tpu"}}, "pretraining/shared"),
        (
            {"pretrained_models": {"shared": {"device": "cpu"}}},
            "pretraining/shared",
        ),
        (
            {"methods": {"alpha": {"classifier_training": "tpu"}}},
            "classifier_training/alpha",
        ),
        ({"methods": {"alpha": {"pretraining": "cpu"}}}, "pretraining"),
        ({"defaults": {"pretraining": "cuda"}}, "GPU allocation"),
        ({"pretrained_models": {"sharde": "cuda"}}, "sharde"),
        (
            {"methods": {"beta": {"classifier_training": "cuda"}}},
            "execution.methods.beta.classifier_training",
        ),
    ],
)
def test_invalid_selected_prerequisite_fails_before_any_submission(
    tmp_path: Path, execution: dict[str, object], diagnostic: str
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["execution"] = execution
    workflow.config["execution_site"] = {
        "cpu": {
            "slurm_partition": "cpu-queue",
            "slurm_account": "cpu-account",
        },
        "gpu": {"gpu": 1, "gres": "gpu:T4:1"},
    }

    result = workflow.run("--executor", "slurm")

    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert workflow.submissions() == []


@pytest.mark.parametrize(
    "rule",
    ["train_classifier", "train_classifier_for_method", "pretrain_model"],
)
def test_prerequisite_resource_conflict_is_rejected_before_upstream_submission(
    tmp_path: Path, rule: str
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["execution_site"] = SITE

    result = workflow.run(
        "--executor", "slurm", "--set-resources", f"{rule}:gpu=1"
    )

    assert result.returncode != 0
    assert "Conflicting allocation" in result.stdout + result.stderr
    assert workflow.submissions() == []


@pytest.mark.pipeline
@pytest.mark.parametrize("profile", ["mixed-gres", "mixed-gpus"])
def test_independent_prerequisites_submit_once_with_matching_script_devices(
    tmp_path: Path, profile: str
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["execution"] = {
        "defaults": {
            "classifier_training": "cuda",
            "pretraining": "cpu",
            "training": "cpu",
            "evaluation": "cpu",
        },
        "methods": {"alpha": {"classifier_training": "cpu"}},
        "pretrained_models": {"shared": "cuda"},
    }

    result = workflow.run(
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles" / profile),
        "--slurm-init-seconds-before-status-checks",
        "0",
        "--seconds-between-status-checks",
        "1",
        "--default-resources",
        "mem_mb=4000",
        "runtime=120",
        "cpus_per_task=1",
        "gpu=4",
        "gres='gpu:wrong:4'",
        "gpu_model='wrong'",
        "--set-resources",
        "pretrain_model:runtime=6000",
        "pretrain_model:cpus_per_task=8",
        "train_classifier:runtime=180",
        "train_classifier:cpus_per_task=2",
        "train_classifier_for_method:runtime=240",
        "train_classifier_for_method:cpus_per_task=4",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    submissions = workflow.submissions()
    assert len(submissions) == 10
    for args in submissions:
        comment = args[args.index("--comment") + 1]
        gpu = "rule_train_classifier_wildcards" in comment or (
            "rule_pretrain_model" in comment and "shared" in comment
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
        gpu_requests = [
            arg for arg in args if arg.startswith(("--gres", "--gpus"))
        ]
        request = (
            "--gres=gpu:T4:1" if profile == "mixed-gres" else "--gpus=a100:1"
        )
        assert gpu_requests == ([request] if gpu else [])
        assert args[args.index("--mem") + 1] == "4000"
        if "rule_pretrain_model" in comment:
            assert args[args.index("-t") + 1] == "6000"
            assert "--cpus-per-task=8" in args
        elif "rule_train_classifier_for_method" in comment:
            assert args[args.index("-t") + 1] == "240"
            assert "--cpus-per-task=4" in args
        elif "rule_train_classifier_wildcards" in comment:
            assert args[args.index("-t") + 1] == "180"
            assert "--cpus-per-task=2" in args
    calls = workflow.script_arguments()
    assert len(calls) == 10
    prerequisites = [
        (script, args)
        for script, args in calls
        if "train_classifier/" in script or "pretrain_model/" in script
    ]
    assert len(prerequisites) == 4
    for script, args in prerequisites:
        gpu = "masked_mlp_classifier.py" in script or "shared.py" in script
        assert args["device"] == ("cuda" if gpu else "cpu")
        assert args["seed"] == "0"
        assert args["smoke_test"] == "True"
        assert (tmp_path / args["save_path"]).is_dir()
    shared = [
        args
        for script, args in calls
        if script.endswith("pretrain_model/shared.py")
    ]
    assert len(shared) == 1
    assert (
        (tmp_path / shared[0]["save_path"])
        .with_name("pretrain_time.txt")
        .is_file()
    )
    for script, args in calls:
        if (
            "train_method/alpha.py" in script
            or "train_method/beta.py" in script
        ):
            assert (
                args["pretrained_model_bundle_path"] == shared[0]["save_path"]
            )
            assert args["device"] == "cpu"


def test_classifier_only_invocation_does_not_resolve_unselected_pretraining(
    tmp_path: Path,
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["execution"] = {
        "defaults": {"classifier_training": "cpu", "pretraining": "cuda"},
        "pretrained_models": {"shared": "tpu"},
    }
    workflow.config["execution_site"] = {
        "cpu": {"slurm_partition": "cpu-queue", "slurm_account": "cpu-account"}
    }

    result = workflow.run("--dry-run", target="all_train_classifiers")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "rule pretrain_model:" not in result.stdout + result.stderr
    assert "device=cpu" in result.stdout + result.stderr


def test_local_cpu_smoke_runs_prerequisites_on_the_same_native_graph(
    tmp_path: Path,
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)

    result = workflow.run("--executor", "local", target="all_train_methods")

    assert result.returncode == 0, result.stdout + result.stderr
    assert workflow.submissions() == []
    calls = workflow.script_arguments()
    assert len(calls) == 7
    for _, args in calls:
        assert args["device"] == "cpu"
        assert args["smoke_test"] == "True"
        assert (tmp_path / args["save_path"]).is_dir()


def test_pretrained_model_choice_uses_model_name_not_script_name(
    tmp_path: Path,
) -> None:
    workflow = prerequisite_workflow(tmp_path)
    add_pretrained_models(workflow)
    workflow.config["pretrain_mapping"] = {
        "shared": {"pretrain_script_name": "shared"},
        "other": {"pretrain_script_name": "shared"},
    }
    workflow.config["execution"] = {
        "defaults": {"pretraining": "cpu"},
        "pretrained_models": {"shared": "cuda"},
    }

    result = workflow.run("--dry-run", target="all_pretrain_models")

    commands = result.stdout + result.stderr
    assert result.returncode == 0, commands
    pretraining = commands.split("python scripts/pretrain_model/shared.py")[1:]
    assert len(pretraining) == 2
    for command in pretraining:
        contract = command.split("END_TIME", 1)[0]
        if "/shared/dataset-cube" in contract:
            assert "device=cuda" in contract
        else:
            assert "/other/dataset-cube" in contract
            assert "device=cpu" in contract
