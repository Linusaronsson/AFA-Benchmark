"""The command the pipeline launcher (`workflow/run.py`) runs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from workflow.run import (
    LauncherError,
    app,
    available_names,
    check_site_venvs,
    launcher_command,
    snakemake_arguments,
    snakemake_command,
)

ESTIMATE_SCRIPT = "scripts/compute_estimate/estimate_compute.py"
REPO_ROOT = Path(__file__).parents[2]


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
    kind: str, expected: str
) -> None:
    result = CliRunner().invoke(app, [f"--list-{kind}"])

    assert result.exit_code == 0, result.output
    assert expected in result.stdout.splitlines()


def test_can_list_multiple_catalogs() -> None:
    result = CliRunner().invoke(
        app, ["--list-datasets", "--list-initializers"]
    )

    assert "Available datasets" in result.stdout
    assert "Available initializers" in result.stdout


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
    catalog.write_text("datasets: miniboone\n")
    with pytest.raises(LauncherError, match="list of names"):
        available_names("datasets", tmp_path)


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_advertises_all_catalogs(flag: str) -> None:
    # Wide enough that no option name is cut off
    result = CliRunner().invoke(app, [flag], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    for kind in ("methods", "datasets", "unmaskers", "initializers"):
        assert f"--list-{kind}" in result.stdout


def test_an_unknown_option_is_refused() -> None:
    with pytest.raises(LauncherError, match="--frobnicate"):
        snakemake_command(["--frobnicate"])


@pytest.mark.parametrize("estimate", [False, True])
def test_launcher_passes_profile_identity_to_the_executing_workflow(
    monkeypatch: pytest.MonkeyPatch,
    estimate: bool,
) -> None:
    monkeypatch.delenv("AFABENCH_PIPELINE_PROFILE", raising=False)
    environments: list[dict[str, str]] = []

    def launch(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == REPO_ROOT
        assert check is False
        environments.append(env)
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr("workflow.run.subprocess.run", launch)

    result = CliRunner().invoke(
        app, ["kdd26", "-e"] if estimate else ["kdd26"]
    )
    assert result.exit_code == 17, result.output
    if estimate:
        assert "AFABENCH_PIPELINE_PROFILE" not in environments[0]
    else:
        assert environments[0]["AFABENCH_PIPELINE_PROFILE"] == (
            "workflow/profiles/pipeline/kdd26"
        )


def test_configfile_replaces_profile_and_keeps_site_and_overrides(
    tmp_path: Path,
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("{}\n")
    command = snakemake_arguments(
        [
            "--configfile",
            str(config),
            "--site",
            "examples/mixed-gres",
            "dataset_realization_indices=[0,1,2]",
            "-n",
        ]
    )
    assert "--profile" not in command
    assert command[:6] == [
        "all",
        "--snakefile",
        "workflow/snakefiles/orchestration/pipeline.smk",
        "--configfile",
        str(config),
        "--workflow-profile",
    ]
    assert "dataset_realization_indices=[0,1,2]" in command
    assert (
        "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml"
        in command
    )
    assert "--dry-run" in command
    assert launcher_command(["--configfile", str(config)]) == [
        "uv",
        "run",
        "python",
        "workflow/run_from_config.py",
        "--configfile",
        str(config),
    ]


def test_configfile_main_clears_inherited_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("{}\n")
    monkeypatch.setenv("AFABENCH_PIPELINE_PROFILE", "publisher-profile")

    def launch(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert "workflow/run_from_config.py" in command
        assert "AFABENCH_PIPELINE_PROFILE" not in env
        assert cwd == REPO_ROOT
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("workflow.run.subprocess.run", launch)
    result = CliRunner().invoke(app, ["--configfile", str(config)])
    assert result.exit_code == 0, result.output


def test_configfile_refuses_named_profile_or_missing_file(
    tmp_path: Path,
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("{}\n")
    with pytest.raises(LauncherError, match="not both"):
        snakemake_arguments(["kdd26", "--configfile", str(config)])
    with pytest.raises(LauncherError, match="No benchmark config"):
        snakemake_arguments(["--configfile", str(tmp_path / "missing.yaml")])


def test_configfile_estimate_keeps_config_input(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("{}\n")
    assert launcher_command(["--configfile", str(config), "-e"]) == [
        "uv",
        "run",
        "python",
        ESTIMATE_SCRIPT,
        "--configfile",
        str(config),
    ]


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


def test_site_naming_venvs_runs_snakemake_from_the_nodes_venv() -> None:
    command = snakemake_command(["--site", "arrhenius"])

    assert command[:3] == ["workflow/venvs/bin/python", "-m", "snakemake"]


def checkout_with_venvs(tmp_path: Path, *, lock: str) -> Path:
    """Copy the site profiles into a checkout with both Arrhenius venvs."""
    shutil.copytree(
        REPO_ROOT / "workflow/profiles", tmp_path / "workflow/profiles"
    )
    (tmp_path / "uv.lock").write_text("# the locked dependencies\n")
    for venv in [".venv-x86_64", ".venv-aarch64"]:
        (tmp_path / venv).mkdir()
        (tmp_path / venv / "afabench.uv.lock").write_text(lock)
    return tmp_path


def test_venvs_built_from_the_lock_pass_the_check(tmp_path: Path) -> None:
    checkout = checkout_with_venvs(
        tmp_path, lock="# the locked dependencies\n"
    )

    check_site_venvs(["--site", "arrhenius"], checkout)


def test_a_stale_or_missing_venv_is_refused_before_launching(
    tmp_path: Path,
) -> None:
    checkout = checkout_with_venvs(tmp_path, lock="# an older lock\n")
    shutil.rmtree(checkout / ".venv-aarch64")

    with pytest.raises(LauncherError) as error:
        check_site_venvs(["--site", "arrhenius", "-n"], checkout)

    assert "No venv built at" in str(error.value)
    assert ".venv-x86_64 was built from another uv.lock" in str(error.value)
    assert "uv run just build-venvs arrhenius" in str(error.value)


def test_sites_without_venvs_and_local_runs_need_no_check(
    tmp_path: Path,
) -> None:
    shutil.copytree(
        REPO_ROOT / "workflow/profiles", tmp_path / "workflow/profiles"
    )

    check_site_venvs([], tmp_path)
    check_site_venvs(["--site", "vera"], tmp_path)


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


@pytest.mark.parametrize(
    ("flag", "override"),
    [
        ("--soft-budget", "soft_budget_setting=true"),
        ("--no-per-dataset-plots", "per_dataset_plots=false"),
        ("--no-wandb", "use_wandb=false"),
    ],
)
def test_each_flag_pair_sets_its_override_either_way(
    flag: str, override: str
) -> None:
    command = snakemake_command([flag])

    assert command[command.index("--config") :] == ["--config", override]


def test_without_flags_the_pipeline_profile_decides() -> None:
    assert "--config" not in snakemake_command([])


def test_wandb_turns_wandb_logging_on_in_a_new_wandb_group() -> None:
    command = snakemake_command(["--wandb", "methods=[dime]"])

    *config, group = command[command.index("--config") :]
    assert config == ["--config", "methods=[dime]", "use_wandb=true"]
    assert re.fullmatch(r"wandb_group=all-\d{8}T\d{6}Z-[a-z0-9]{4}", group)


def test_each_invocation_gets_a_new_wandb_group() -> None:
    first, second = (
        snakemake_command(["--wandb", "kdd26"])[-1] for _ in range(2)
    )

    assert first.startswith("wandb_group=kdd26-")
    assert first != second


@pytest.mark.parametrize("smoke_test", ["smoke_test=true", "smoke_test=True"])
def test_a_smoke_test_wandb_group_says_so(smoke_test: str) -> None:
    command = snakemake_command(["--wandb", smoke_test])

    assert command[-1].endswith("-smoke")


def test_a_wandb_logging_override_gets_a_new_wandb_group_too() -> None:
    command = snakemake_command(["use_wandb=true"])

    assert command[-1].startswith("wandb_group=all-")


def test_wandb_group_continues_a_named_wandb_group() -> None:
    command = snakemake_command(["--wandb-group", "all-x", "methods=[dime]"])

    assert command[command.index("--config") :] == [
        "--config",
        "methods=[dime]",
        "use_wandb=true",
        "wandb_group=all-x",
    ]


def test_a_wandb_group_override_is_kept() -> None:
    command = snakemake_command(["--wandb", "wandb_group=all-x"])

    assert command[command.index("--config") :] == [
        "--config",
        "wandb_group=all-x",
        "use_wandb=true",
    ]


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (["--wandb-group", "all-x", "wandb_group=all-y"], "--wandb-group"),
        (["--wandb-group", "all-x", "use_wandb=false"], "--wandb-group"),
        (["--wandb-group", ""], "name"),
        (["--no-wandb", "--wandb-group", "all-x"], "--no-wandb"),
    ],
)
def test_wandb_group_rejects_contradictions(
    arguments: list[str], match: str
) -> None:
    with pytest.raises(LauncherError, match=match):
        snakemake_command(arguments)


def test_wandb_logging_is_off_without_wandb() -> None:
    assert not any(
        argument.startswith(("use_wandb", "wandb_group"))
        for argument in snakemake_command(["methods=[dime]"])
    )


def test_wandb_and_its_override_exclude_each_other() -> None:
    with pytest.raises(LauncherError, match="--wandb"):
        snakemake_command(["--wandb", "use_wandb=false"])


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


def test_estimate_on_a_site_naming_venvs_runs_in_the_nodes_venv() -> None:
    command = launcher_command(
        ["--site", "arrhenius", "-e", "datasets=[cube]"]
    )

    assert command == [
        "workflow/venvs/bin/python",
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


def test_per_dataset_plots_opt_in_reaches_the_site_config() -> None:
    command = snakemake_command(
        ["--per-dataset-plots", "-n", "--site", "examples/mixed-gres"]
    )

    assert command[command.index("--config") :] == [
        "--config",
        "per_dataset_plots=true",
        "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml",
    ]
    assert "--dry-run" in command


def test_per_dataset_plots_and_its_override_exclude_each_other() -> None:
    with pytest.raises(LauncherError, match="--per-dataset-plots"):
        snakemake_command(["--per-dataset-plots", "per_dataset_plots=false"])


def test_per_dataset_plots_is_off_without_the_flag() -> None:
    assert not any(
        argument.startswith("per_dataset_plots=")
        for argument in snakemake_command([])
    )


def test_estimate_passes_per_dataset_plots_on() -> None:
    assert launcher_command(["-e", "--per-dataset-plots"])[-1] == (
        "--per-dataset-plots"
    )


def test_interactive_preview_browses_the_runs_snakemake_arguments() -> None:
    command = launcher_command(
        [
            "--interactive-preview",
            "methods=[dime]",
            "--target",
            "all_train_methods",
        ]
    )

    assert command == [
        "uv",
        "run",
        "python",
        "-m",
        "afabench.pipeline_preview",
        "--",
        *snakemake_arguments(
            ["methods=[dime]", "--target", "all_train_methods"]
        ),
    ]


def test_interactive_preview_of_a_config_file_plans_instead_of_running_it(
    tmp_path: Path,
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("{}\n")

    command = launcher_command(
        ["--configfile", str(config), "--interactive-preview"]
    )

    # Not the config invocation's phases (workflow/run_from_config.py): the
    # planner prepares a release preview itself, for planning only.
    assert command == [
        "uv",
        "run",
        "python",
        "-m",
        "afabench.pipeline_preview",
        "--",
        *snakemake_arguments(["--configfile", str(config)]),
    ]
    assert not any("release_preview" in argument for argument in command)


def test_interactive_preview_on_a_site_naming_venvs_runs_in_the_nodes_venv() -> (
    None
):
    command = launcher_command(
        ["--site", "arrhenius", "--interactive-preview", "datasets=[cube]"]
    )

    assert command[:5] == [
        "workflow/venvs/bin/python",
        "-m",
        "afabench.pipeline_preview",
        "--",
        "--profile",
    ]
    assert (
        "execution_site_file=workflow/profiles/site/arrhenius/site.yaml"
        in (command)
    )


def test_interactive_preview_implies_a_dry_run_so_dash_n_changes_nothing() -> (
    None
):
    assert launcher_command(["--interactive-preview", "-n", "kdd26"]) == (
        launcher_command(["--interactive-preview", "kdd26"])
    )


def test_dash_i_opens_the_interactive_preview() -> None:
    assert launcher_command(["-i", "--group-by", "method", "kdd26"]) == (
        launcher_command(
            ["--interactive-preview", "--group-by", "method", "kdd26"]
        )
    )


@pytest.mark.parametrize(
    ("option", "named"),
    [
        (["-e"], "-e"),
        (["--estimate"], "-e"),
        (["--wandb"], "--wandb"),
        (["--wandb-group", "earlier-run"], "--wandb-group"),
    ],
)
def test_interactive_preview_rejects_estimate_and_wandb_logging(
    option: list[str], named: str
) -> None:
    with pytest.raises(LauncherError, match=re.escape(named)):
        launcher_command(["--interactive-preview", *option])


def test_interactive_preview_needs_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[list[str]] = []

    def launch(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        launched.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("workflow.run.subprocess.run", launch)
    # The test runner's streams are not a terminal.
    result = CliRunner().invoke(app, ["--interactive-preview"])
    assert result.exit_code == 2
    assert launched == []
    assert "terminal" in result.stderr
    assert "-n" in result.stderr


def test_group_by_sets_the_interactive_previews_initial_grouping() -> None:
    command = launcher_command(
        ["--interactive-preview", "--group-by", "dataset,method", "kdd26"]
    )

    assert command[3:8] == [
        "-m",
        "afabench.pipeline_preview",
        "--group-by",
        "dataset,method",
        "--",
    ]
    assert command[8:] == snakemake_arguments(["kdd26"])


@pytest.mark.parametrize(
    "grouping", ["stage", "method,dataset", "dataset,stage,method"]
)
def test_group_by_takes_any_ordered_subset_of_the_dimensions(
    grouping: str,
) -> None:
    command = launcher_command(
        ["--interactive-preview", "--group-by", grouping]
    )

    assert command[command.index("--group-by") + 1] == grouping


@pytest.mark.parametrize(
    ("grouping", "named"),
    [
        ("", "''"),
        ("stage,budget", "'budget'"),
        ("method,stage,method", "'method'"),
        ("stage,,method", "''"),
    ],
)
def test_group_by_names_known_dimensions_once(
    grouping: str, named: str
) -> None:
    with pytest.raises(LauncherError, match=re.escape(named)):
        launcher_command(["--interactive-preview", "--group-by", grouping])


def test_group_by_needs_interactive_preview() -> None:
    with pytest.raises(LauncherError, match="--interactive-preview"):
        launcher_command(["--group-by", "method"])
    with pytest.raises(LauncherError, match="--interactive-preview"):
        launcher_command(["-n", "--group-by", "method"])
