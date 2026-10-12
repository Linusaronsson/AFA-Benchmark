"""The command the pipeline launcher (`workflow/run.py`) runs."""

from pathlib import Path

import pytest

from workflow.run import (
    LauncherError,
    available_names,
    launcher_command,
    snakemake_arguments,
    snakemake_command,
)

ESTIMATE_SCRIPT = "scripts/compute_estimate/estimate_compute.py"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("methods", "l2m_synthetic_feature_prior"),
        ("datasets", "miniboone"),
        ("unmaskers", "28x28_to_4x4"),
        ("initializers", "warm"),
    ],
)
def test_lists_shipped_names_without_launching(
    kind: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        launcher_command([f"--list-{kind}"])
    assert exit_info.value.code == 0
    assert expected in capsys.readouterr().out.splitlines()


def test_can_list_multiple_catalogs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        launcher_command(["--list-datasets", "--list-initializers"])
    output = capsys.readouterr().out
    assert "Available datasets" in output
    assert "Available initializers" in output


def test_catalog_reads_another_checkout(tmp_path: Path) -> None:
    catalog = tmp_path / "workflow/conf/methods/all.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "# custom catalog\nmethods:\n  - new_method # comment\n"
    )
    assert available_names("methods", tmp_path) == ["new_method"]


def test_catalog_rejects_unsupported_yaml(tmp_path: Path) -> None:
    catalog = tmp_path / "workflow/conf/datasets/all.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("datasets: [miniboone]\n")
    with pytest.raises(LauncherError, match="flat list"):
        available_names("datasets", tmp_path)


def test_help_advertises_all_catalogs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        launcher_command(["--help"])
    output = capsys.readouterr().out
    for kind in ("methods", "datasets", "unmaskers", "initializers"):
        assert f"--list-{kind}" in output


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


def test_no_soft_budget_turns_the_soft_budget_setting_off() -> None:
    command = snakemake_command(
        ["--no-soft-budget", "methods=[dime]", "--site", "examples/mixed-gres"]
    )

    assert command[command.index("--config") :] == [
        "--config",
        "methods=[dime]",
        "soft_budget_setting=false",
        "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml",
    ]


def test_no_soft_budget_and_its_override_exclude_each_other() -> None:
    with pytest.raises(LauncherError, match="--no-soft-budget"):
        snakemake_command(["--no-soft-budget", "soft_budget_setting=true"])


def test_estimate_passes_no_soft_budget_on() -> None:
    command = launcher_command(["-e", "--no-soft-budget"])

    assert command[-1] == "--no-soft-budget"


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


def test_estimate_takes_the_arguments_of_the_run() -> None:
    command = launcher_command(
        ["-e", "methods=[dime]", "--target", "all_train_methods"]
    )

    assert command == [
        "uv",
        "run",
        "python",
        ESTIMATE_SCRIPT,
        "methods=[dime]",
        "--target",
        "all_train_methods",
    ]


def test_estimate_on_a_site_whose_jobs_run_in_an_image_runs_in_the_image() -> (
    None
):
    command = launcher_command(
        ["--site", "arrhenius", "-e", "datasets=[cube]"]
    )

    assert command == [
        "containers/run.sh",
        "python",
        ESTIMATE_SCRIPT,
        "--site",
        "arrhenius",
        "datasets=[cube]",
    ]


def test_estimate_options_and_snakemake_arguments_reach_the_estimate() -> None:
    command = launcher_command(
        [
            "--estimate",
            "--job-durations",
            "durations.parquet",
            "--by",
            "name",
            "--by",
            "dataset_key",
            "--output",
            "estimate.csv",
            "--strict",
            "--",
            "--set-resources",
            "train_method:gpu=1",
        ]
    )

    assert command[4:] == [
        "--job-durations",
        "durations.parquet",
        "--by",
        "name",
        "--by",
        "dataset_key",
        "--output",
        "estimate.csv",
        "--strict",
        "--",
        "--set-resources",
        "train_method:gpu=1",
    ]


def test_without_estimate_the_launcher_runs_snakemake() -> None:
    assert launcher_command(["-n"]) == snakemake_command(["-n"])


def test_estimate_and_dry_run_exclude_each_other() -> None:
    with pytest.raises(LauncherError, match=r"-n.*-e"):
        launcher_command(["-n", "-e"])


@pytest.mark.parametrize(
    "option",
    [["--job-durations", "x.parquet"], ["--by", "name"], ["--strict"]],
)
def test_estimate_options_need_estimate(option: list[str]) -> None:
    with pytest.raises(LauncherError, match=option[0]):
        launcher_command(option)
