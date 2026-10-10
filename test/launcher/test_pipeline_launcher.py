"""The Snakemake command the pipeline launcher (`workflow/run.py`) runs."""

from pathlib import Path

import pytest

from workflow.run import LauncherError, snakemake_arguments, snakemake_command


def test_runs_the_all_pipeline_profile_locally_by_default() -> None:
    assert snakemake_command([]) == [
        "uv",
        "run",
        "snakemake",
        "--profile",
        "workflow/profiles/pipeline/all",
        "all",
    ]


def test_site_and_overrides_share_one_config() -> None:
    command = snakemake_command(
        [
            "kdd26",
            "--site",
            "examples/mixed-gres",
            "methods=[dime,gdfs]",
            "eval_hard_budgets={cube: [2, 4]}",
        ]
    )

    assert command == [
        "uv",
        "run",
        "snakemake",
        "--profile",
        "workflow/profiles/pipeline/kdd26",
        "--workflow-profile",
        "workflow/profiles/site/examples/mixed-gres",
        "all",
        "--config",
        "methods=[dime,gdfs]",
        "eval_hard_budgets={cube: [2, 4]}",
        "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml",
    ]


def test_site_whose_jobs_run_in_an_image_uses_the_host_environment() -> None:
    command = snakemake_command(["--site", "arrhenius"])

    assert command[:3] == ["containers/bin/python", "-m", "snakemake"]


def test_dry_run_target_and_extra_snakemake_arguments() -> None:
    command = snakemake_command(
        [
            "-n",
            "--target",
            "all_train_methods",
            "methods=[dime]",
            "--",
            "--rerun-incomplete",
            "--keep-going",
        ]
    )

    assert command == [
        "uv",
        "run",
        "snakemake",
        "--profile",
        "workflow/profiles/pipeline/all",
        "all_train_methods",
        "--dry-run",
        "--printshellcmds",
        "--rerun-incomplete",
        "--keep-going",
        "--config",
        "methods=[dime]",
    ]


def test_local_run_takes_a_job_count() -> None:
    command = snakemake_command(["--jobs", "8"])

    assert command[-3:] == ["all", "--jobs", "8"]


def test_cluster_run_takes_its_job_count_from_the_site_profile() -> None:
    with pytest.raises(LauncherError, match="--jobs"):
        snakemake_command(["--site", "examples/mixed-gres", "--jobs", "8"])


def test_unknown_pipeline_profile_lists_the_shipped_ones() -> None:
    with pytest.raises(LauncherError, match=r"'al'.*all, kdd26"):
        snakemake_command(["al"])


def test_unknown_site_profile_lists_the_shipped_ones() -> None:
    with pytest.raises(LauncherError, match=r"'arrenius'.*arrhenius"):
        snakemake_command(["--site", "arrenius"])


def test_a_word_after_the_pipeline_profile_must_be_an_override() -> None:
    with pytest.raises(LauncherError, match=r"'\[dime\]'"):
        snakemake_command(["methods", "[dime]"])


def test_arguments_resolve_profiles_in_another_checkout(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "workflow/profiles/pipeline/mine"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("snakefile: Snakefile\n")

    arguments = snakemake_arguments(["mine", "-n"], checkout=tmp_path)

    assert arguments == [
        "--profile",
        "workflow/profiles/pipeline/mine",
        "all",
        "--dry-run",
        "--printshellcmds",
    ]


def test_overrides_cannot_bypass_the_launcher_after_the_separator() -> None:
    # A second --config would replace the one carrying the site file.
    with pytest.raises(LauncherError, match="--config"):
        snakemake_command(
            ["--site", "examples/mixed-gres", "--", "--config", "a=1"]
        )


def test_a_site_file_without_allocations_is_refused(tmp_path: Path) -> None:
    for profiles, name in [("pipeline", "all"), ("site", "broken")]:
        profile = tmp_path / "workflow/profiles" / profiles / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("jobs: 1\n")
    (tmp_path / "workflow/profiles/site/broken/site.yaml").write_text(
        "cpu:\n  slurm_partition: cpu\n"
    )

    with pytest.raises(LauncherError, match="execution_site"):
        snakemake_arguments(["--site", "broken"], checkout=tmp_path)
