"""CPU-only processing through the real orchestration command boundary."""

import shutil
from pathlib import Path

import pytest

from test.workflow.submission_harness import SITE, WorkflowHarness


@pytest.mark.parametrize(
    ("dataset", "script"),
    [
        ("cube", "generate_dataset.py"),
        ("imagenette", "generate_image_dataset.py"),
    ],
)
def test_dataset_generation_clears_gpu_defaults(
    tmp_path: Path, dataset: str, script: str
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["datasets"] = [dataset]
    shutil.rmtree(tmp_path / "extra/output_smoke/datasets")
    workflow.config["execution_site"] = {
        "cpu": {
            "slurm_partition": "cpu-queue",
            "slurm_account": "cpu-account",
        },
    }

    result = workflow.run(
        "--dry-run",
        "--default-resources",
        "gpu=2",
        "gres=gpu:T4:2",
        "gpu_model=T4",
        "slurm_partition=gpu-queue",
        target="all_generate_datasets",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = (
        (result.stdout + result.stderr)
        .split("rule dataset_generation:", 1)[1]
        .split("rule all_generate_datasets:", 1)[0]
    )
    assert "slurm_partition=cpu-queue" in commands
    assert "slurm_account=cpu-account" in commands
    assert "gpu=0" in commands
    assert "gpu=2" not in commands
    assert "gres=gpu:" not in commands
    assert f"scripts/dataset_generation/{script}" in commands
    assert "dataset_realization_indices=[0]" in commands
    assert f"save_path=extra/output_smoke/datasets/{dataset}" in commands


CPU_RULES = {
    "dataset_generation",
    "transform_eval_data",
    "merge_eval_perf",
    "split_by_classifier_type",
    "collect_job_records",
    "plot_eval_perf",
    "plot_eval_actions",
    "plot_time",
}


def processing_workflow(root: Path) -> WorkflowHarness:
    workflow = WorkflowHarness(root)
    shutil.rmtree(root / "extra/output_smoke/datasets")
    workflow.config.update(
        {
            "execution": {
                "defaults": {"training": "cuda", "evaluation": "cuda"},
            },
            "method_options": {
                "alpha": {
                    "train_script_name": "alpha",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                },
                "beta": {"train_script_name": "beta", "eval_batch_size": 1},
            },
            "pretrain_mapping": {"shared": {"pretrain_script_name": "shared"}},
            "method_sets": {"heatmap_comparison": ["alpha", "beta"]},
        }
    )
    pretrained = (
        root / "extra/output_smoke/pretrained_models/initializer-cold/shared/"
        "dataset-cube+realization_index-0/pretrain_seed-0"
    )
    (pretrained / "model.bundle").mkdir(parents=True)
    for script in [
        "pretrain_model/shared.py",
        "train_classifier/masked_mlp_classifier.py",
    ]:
        path = root / "scripts" / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((root / "scripts/train_method/alpha.py").read_text())
    # Replace expensive scripts only, retaining all production rule wiring.
    for script in [
        "dataset_generation/generate_dataset.py",
        "misc/transform_eval_data_pipeline.py",
        "misc/merge_dataframes.py",
        "misc/split_eval_perf_by_classifier.py",
        "plotting/plot_eval_perf.py",
        "plotting/plot_eval_actions.py",
        "plotting/plot_total_time.py",
    ]:
        path = root / "scripts" / script
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = {}\n"
            "argv = iter(sys.argv[1:])\n"
            "for arg in argv:\n"
            "    if arg.startswith('--'): args[arg[2:]] = next(argv)\n"
            "    elif '=' in arg: key, value = arg.split('=', 1); args[key] = value\n"
            "with Path(os.environ['ARGUMENTS']).open('a') as f:\n"
            "    f.write(json.dumps([sys.argv[0], args]) + '\\n')\n"
            "if 'dataset_realization_indices' in args:\n"
            "    for index in args['dataset_realization_indices'].strip('[]').split(','):\n"
            "        for split in ['train', 'val', 'test']:\n"
            "            (Path(args['save_path']) / index / (split + '.bundle')).mkdir(parents=True, exist_ok=True)\n"
            "for key in ['output', 'output_path', 'output_builtin', 'output_external']:\n"
            "    if key in args:\n"
            "        out = Path(args[key]); out.parent.mkdir(parents=True, exist_ok=True)\n"
            "        out.write_text('fixture')\n"
            "if 'output_folder' in args:\n"
            "    out = Path(args['output_folder']); out.mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'fixture.svg').write_text('<svg/>')\n"
        )
    return workflow


@pytest.mark.parametrize("legacy", [False, True])
def test_full_graph_processing_is_cpu_only(
    tmp_path: Path, *, legacy: bool
) -> None:
    workflow = processing_workflow(tmp_path)
    if legacy:
        del workflow.config["execution"]
        workflow.config["device"] = "cuda"

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles/mixed-gres"),
        "--default-resources",
        "gpu=2",
        "gres=gpu:T4:2",
        "gpu_model=T4",
        target="all",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout + result.stderr
    for rule in CPU_RULES:
        block = commands.split(f"rule {rule}:", 1)[1].split(
            "Shell command:", 1
        )[0]
        assert "slurm_partition=cpu-queue" in block, block
        assert "slurm_account=cpu-account" in block, block
        assert "gpu=0" in block, block
        assert "gres=gpu:" not in block, block
    assert commands.count("rule train_method:") == 2
    assert commands.count("rule eval_method:") == 2
    assert "device=cuda" in commands


@pytest.mark.parametrize("rule", sorted(CPU_RULES))
def test_cpu_only_rule_rejects_gpu_override_before_submission(
    tmp_path: Path, rule: str
) -> None:
    workflow = processing_workflow(tmp_path)
    workflow.config["execution_site"] = SITE

    result = workflow.run(
        "--executor",
        "slurm",
        "--set-resources",
        f"{rule}:gpu=1",
        target="all",
    )

    assert result.returncode != 0
    assert "Conflicting allocation" in result.stdout + result.stderr
    assert workflow.submissions() == []


@pytest.mark.parametrize("slurm_extra", ["--gpus=2", "--qos=short"])
def test_profile_default_slurm_extra_is_rejected_before_submission(
    tmp_path: Path, slurm_extra: str
) -> None:
    # Every job's allocation replaces it, so accepting it would drop it.
    workflow = processing_workflow(tmp_path)
    workflow.config["execution_site"] = SITE

    result = workflow.run(
        "--executor",
        "slurm",
        "--default-resources",
        f"slurm_extra='{slurm_extra}'",
        target="all",
    )

    assert result.returncode != 0
    assert "default-resources slurm_extra" in result.stdout + result.stderr
    assert workflow.submissions() == []


@pytest.mark.pipeline
@pytest.mark.parametrize("profile", ["mixed-gres", "mixed-gpus"])
def test_cpu_processing_submissions_clear_site_gpu_defaults(
    tmp_path: Path, profile: str
) -> None:
    workflow = processing_workflow(tmp_path)

    result = workflow.run(
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles" / profile),
        "--default-resources",
        "runtime=135",
        "mem_mb=4500",
        "cpus_per_task=3",
        "slurm_partition=gpu-default",
        "slurm_account=gpu-account",
        "gpu=2",
        "gres=gpu:T4:2",
        "gpu_model=T4",
        "--set-resources",
        "plot_eval_perf:cpus_per_task=10",
        "split_by_classifier_type:cpus_per_task=10",
        "--slurm-init-seconds-before-status-checks",
        "0",
        "--seconds-between-status-checks",
        "1",
        target="all",
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    seen = set()
    submissions = workflow.submissions()
    assert len(submissions) == 16
    for args in submissions:
        comment = args[args.index("--comment") + 1]
        gpu = "rule_train_method" in comment or "rule_eval_method" in comment
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
        if gpu:
            assert (
                "--gres=gpu:T4:1"
                if profile == "mixed-gres"
                else "--gpus=a100:1"
            ) in args
            continue
        if any(
            f"rule_{rule}_" in comment
            for rule in ["train_classifier", "pretrain_model"]
        ):
            continue
        rule = next(
            rule
            for rule in CPU_RULES
            if comment == f"rule_{rule}"
            or comment.startswith(f"rule_{rule}_wildcards_")
        )
        seen.add(rule)
        assert not any(arg.startswith(("--gpus", "--gres")) for arg in args)
        assert args[args.index("-t") + 1] == "135"
        assert args[args.index("--mem") + 1] == "4500"
        cpus = (
            10 if rule in {"plot_eval_perf", "split_by_classifier_type"} else 3
        )
        assert f"--cpus-per-task={cpus}" in args
    assert seen == CPU_RULES
    # Every job but collect_job_records, which runs for real, runs a stub.
    calls = workflow.script_arguments()
    assert len(calls) == 15
    for script, args in calls:
        if "train_method" in script or "eval/" in script:
            assert args["device"] == "cuda"
        elif "dataset_generation" in script:
            assert args["dataset_realization_indices"] == "[0]"
            assert args["seeds"] == "[0]"
            assert args["save_path"] == "extra/output_smoke/datasets/cube"
        elif "plotting" in script:
            assert args["formats"] == "[pdf,svg]"
            assert (tmp_path / args["output_folder"] / "fixture.svg").is_file()
            if "plot_total_time" in script:
                assert args == {
                    "input": "extra/output_smoke/merged_results/job_duration_table.parquet",
                    "output_folder": "extra/output_smoke/plot_results/eval_split-test/initializer-cold/time",
                    "methods": "[alpha,beta]",
                    "++pretrained_models": "{alpha:shared}",
                    "initializer_tag": "initializer-cold",
                    "eval_dataset_split": "test",
                    "formats": "[pdf,svg]",
                }
        elif "transform_eval" in script:
            assert args["dataset"] == "cube"
            assert args["initializer"] == "cold"
            assert (tmp_path / args["output_path"]).is_file()
    assert (
        tmp_path / "extra/output_smoke/datasets/cube/0/test.bundle"
    ).is_dir()
    assert (
        tmp_path
        / "extra/output_smoke/merged_results/job_duration_table.parquet"
    ).is_file()


@pytest.mark.parametrize("variant", ["pipeline_no_train", "pipeline_no_eval"])
def test_processing_variants_use_cpu_site_mapping(
    tmp_path: Path, variant: str
) -> None:
    workflow = processing_workflow(tmp_path)
    for split in ["train", "val", "test"]:
        (
            tmp_path / f"extra/output_smoke/datasets/cube/0/{split}.bundle"
        ).mkdir(parents=True)
    for method, pretrain in [
        ("alpha", "pretrain_seed-0"),
        ("beta", "NO_PRETRAIN"),
    ]:
        training = (
            f"{method}/dataset-cube+realization_index-0/{pretrain}/"
            "train_seed-0+train_hard_budget-1+train_soft_budget_param-null"
        )
        evaluation = (
            f"{training}/eval_seed-0+eval_hard_budget-1+"
            "eval_soft_budget_param-null"
        )
        trained = (
            tmp_path
            / f"extra/output_smoke/trained_methods/initializer-cold/{training}"
        )
        (trained / "method.bundle").mkdir(parents=True)
        eval_table = (
            tmp_path / "extra/output_smoke/eval_results/eval_split-test/"
            f"initializer-cold/{evaluation}/eval_data.parquet"
        )
        eval_table.parent.mkdir(parents=True, exist_ok=True)
        eval_table.write_text("fixture")

    result = workflow.run(
        "--dry-run",
        "--snakefile",
        f"extra/workflow/snakefiles/orchestration/{variant}.smk",
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles/mixed-gres"),
        "--default-resources",
        "gpu=1",
        target="all",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout + result.stderr
    for rule in CPU_RULES - {"dataset_generation"}:
        block = commands.split(f"rule {rule}:", 1)[1].split(
            "Shell command:", 1
        )[0]
        assert "slurm_partition=cpu-queue" in block, block
        assert "gpu=0" in block, block
    assert "rule train_method:" not in commands
    assert "rule dataset_generation:" not in commands


@pytest.mark.pipeline
def test_local_cpu_processing_reaches_native_final_outputs(
    tmp_path: Path,
) -> None:
    workflow = processing_workflow(tmp_path)
    workflow.config["execution"] = {
        "defaults": {"training": "cpu", "evaluation": "cpu"},
    }

    result = workflow.run("--executor", "local", target="all")

    assert result.returncode == 0, result.stdout + result.stderr
    assert workflow.submissions() == []
    calls = workflow.script_arguments()
    assert len(calls) == 15
    for _, args in calls:
        if "device" in args:
            assert args["device"] == "cpu"
    plots = (
        tmp_path
        / "extra/output_smoke/plot_results/eval_split-test/initializer-cold"
    )
    assert len(list(plots.rglob("fixture.svg"))) == 4
