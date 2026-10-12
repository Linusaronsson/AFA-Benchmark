"""Opt-in figures through the real launcher and Snakemake DAG boundary."""

from pathlib import Path

from test.workflow.submission_harness import job_counts
from test.workflow.test_cpu_processing_execution import (
    WITHOUT_EVALUATION,
    trained_workflow,
)


def test_per_dataset_plots_are_owned_once_per_method_set(
    tmp_path: Path,
) -> None:
    workflow = trained_workflow(tmp_path, evaluated=True)
    workflow.config["stages"] = WITHOUT_EVALUATION
    # Write the harness profile before using the public launcher interface.
    baseline = workflow.run("--dry-run", target="all")
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr

    result = workflow.run_launcher(
        ["harness", "--per-dataset-plots", "-n", "--jobs", "2"]
    )

    commands = result.stdout + result.stderr
    assert result.returncode == 0, commands
    counts = job_counts(commands)
    assert counts["plot_eval_perf"] == 2
    assert counts["plot_eval_perf_per_dataset"] == 2
    assert commands.count("per_dataset_plots=true") == 2
    for method_set in ["all", "heatmap_comparison"]:
        assert (
            f"per_dataset_input=output/smoke/merged_results/eval_split-test/"
            f"initializer-cold/eval_perf/method_set-{method_set}+all.parquet"
        ) in commands
        assert (
            "per_dataset_output_folder=output/smoke/plots/eval_split-test/"
            f"initializer-cold/eval_perf_per_dataset/method_set-{method_set}"
        ) in commands
    assert "  per-dataset plots: True" in commands


def test_plotting_metric_choices_reach_both_usual_classifier_jobs(
    tmp_path: Path,
) -> None:
    workflow = trained_workflow(tmp_path, evaluated=True)
    workflow.config["stages"] = WITHOUT_EVALUATION
    workflow.run("--dry-run", target="all")

    result = workflow.run_launcher(
        [
            "harness",
            "--per-dataset-plots",
            "-n",
            "plotting={dataset_metrics: {cube: auroc}, default_metric: f1}",
        ]
    )

    commands = result.stdout + result.stderr
    assert result.returncode == 0, commands
    assert commands.count("++plotting.dataset_metrics={cube:auroc}") == 4
    assert commands.count("plotting.default_metric=f1") == 4


def test_enabling_figures_after_a_cached_default_run_tracks_new_outputs(
    tmp_path: Path,
) -> None:
    workflow = trained_workflow(tmp_path, evaluated=True)
    workflow.config["stages"] = WITHOUT_EVALUATION
    # Scripts are subprocess boundaries; retain the real production DAG.
    script = tmp_path / "scripts/plotting/plot_eval_perf.py"
    with script.open("a") as stream:
        stream.write(
            "if args.get('per_dataset_plots') == 'true':\n"
            "    out = Path(args['per_dataset_output_folder']) / 'dataset-cube'\n"
            "    out.mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'hard_budget_normal.svg').write_text('<svg/>')\n"
        )
    baseline = workflow.run("--executor", "local", target="all")
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    baseline_commands = baseline.stdout + baseline.stderr
    assert "per_dataset_plots=true" not in baseline_commands
    assert "plot_eval_perf_per_dataset" not in baseline_commands
    plots = tmp_path / "output/smoke/plots/eval_split-test/initializer-cold"
    assert not (plots / "eval_perf_per_dataset").exists()

    enabled = workflow.run_launcher(
        ["harness", "--per-dataset-plots", "-n", "--jobs", "2"]
    )
    commands = enabled.stdout + enabled.stderr
    assert enabled.returncode == 0, commands
    assert job_counts(commands) == {
        "all": 1,
        "plot_eval_perf_per_dataset": 2,
        "total": 3,
    }
    completed = workflow.run_launcher(
        ["harness", "--per-dataset-plots", "--jobs", "2"]
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sorted(
        path.relative_to(plots).as_posix()
        for path in (plots / "eval_perf_per_dataset").rglob("*.svg")
    ) == [
        "eval_perf_per_dataset/method_set-all/dataset-cube/hard_budget_normal.svg",
        "eval_perf_per_dataset/method_set-heatmap_comparison/dataset-cube/hard_budget_normal.svg",
    ]
    cached = workflow.run_launcher(
        ["harness", "--per-dataset-plots", "-n", "--jobs", "2"]
    )
    assert cached.returncode == 0, cached.stdout + cached.stderr
    assert "Nothing to be done" in cached.stdout + cached.stderr

    changed_metrics = workflow.run_launcher(
        [
            "harness",
            "--per-dataset-plots",
            "-n",
            "--jobs",
            "2",
            "plotting={dataset_metrics: {cube: auroc}}",
        ]
    )
    assert changed_metrics.returncode == 0, (
        changed_metrics.stdout + changed_metrics.stderr
    )
    assert job_counts(changed_metrics.stdout + changed_metrics.stderr) == {
        "all": 1,
        "plot_eval_perf": 2,
        "plot_eval_perf_per_dataset": 2,
        "total": 5,
    }


def test_optional_output_cannot_alias_a_usual_classifier_output(
    tmp_path: Path,
) -> None:
    workflow = trained_workflow(tmp_path, evaluated=True)
    workflow.config["stages"] = WITHOUT_EVALUATION
    plots = tmp_path / "output/smoke/plots/eval_split-test/initializer-cold"
    usual = plots / "eval_perf/method_set-all+classifier_type-builtin"
    usual.mkdir(parents=True)
    optional = plots / "eval_perf_per_dataset/method_set-all"
    optional.parent.mkdir(parents=True)
    optional.symlink_to(usual, target_is_directory=True)
    workflow.config["per_dataset_plots"] = True

    result = workflow.run("--dry-run", target="all")

    assert result.returncode != 0
    assert "overlap" in result.stdout + result.stderr
    assert "method_set-all+classifier_type-builtin" in (
        result.stdout + result.stderr
    )
