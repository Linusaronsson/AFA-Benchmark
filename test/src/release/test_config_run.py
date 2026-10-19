"""One config invocation prepares data and validates release reuse in order."""

import subprocess
from pathlib import Path

import pytest
import yaml

from afabench.release.config_run import config_run_commands, run_config
from afabench.release.workflow_config import resolve_workflow_config
from workflow.run import snakemake_arguments


@pytest.fixture
def benchmark(tmp_path: Path) -> Path:
    record = resolve_workflow_config(
        profile=Path("workflow/profiles/pipeline/all"),
        configfiles=[],
        overrides=[],
    )
    config = {
        **record.benchmark,
        "output_root": str(tmp_path / "output"),
        "datasets": ["cube"],
        "methods": ["random_dummy"],
        "dataset_realization_indices": [0, 1],
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def arguments(benchmark: Path, *overrides: str) -> list[str]:
    return snakemake_arguments(["--configfile", str(benchmark), *overrides])


def cache(benchmark: Path) -> Path:
    root = benchmark.parent / "output"
    root.mkdir()
    (root / "release_manifest.json").write_text("{}")
    return root


def test_config_without_release_is_an_ordinary_invocation(
    benchmark: Path,
) -> None:
    command = arguments(benchmark)
    assert config_run_commands(command) == [command]


@pytest.mark.parametrize("site", [[], ["--site", "examples/mixed-gres"]])
def test_release_prepares_datasets_then_enables_checked_reuse(
    benchmark: Path,
    site: list[str],
) -> None:
    cache(benchmark)
    prepare, execute = config_run_commands(
        arguments(
            benchmark,
            "stages={training: false}",
            "dataset_realization_indices=[0,1,2]",
            *site,
        )
    )
    assert prepare[0] == "all_generate_datasets"
    assert "reuse_release=false" in prepare
    if site:
        for phase in (prepare, execute):
            assert "--workflow-profile" in phase
            assert (
                "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml"
                in phase
            )
    assert "dataset_realization_indices=[0,1,2]" in prepare
    assert execute[0] == "all"
    record = resolve_workflow_config(
        profile=None,
        configfiles=[benchmark],
        overrides=execute[execute.index("--config") + 1 :],
    )
    assert record.merged["stages"] == {
        "dataset_generation": "false",
        "training": "false",
        "evaluation": "true",
    }
    assert record.merged["reuse_release"] == "true"


def test_disabling_dataset_generation_skips_preparation(
    benchmark: Path,
) -> None:
    cache(benchmark)
    commands = config_run_commands(
        arguments(benchmark, "stages={dataset_generation: false}")
    )
    assert len(commands) == 1
    assert "reuse_release=true" in commands[0]


def test_dry_run_and_estimate_only_preview_and_never_prepare(
    benchmark: Path,
) -> None:
    cache(benchmark)
    commands = config_run_commands(arguments(benchmark), preview=True)
    assert len(commands) == 1
    assert "--dry-run" in commands[0]
    assert "all_generate_datasets" in commands[0]
    assert "release_preview=true" in commands[0]
    assert "reuse_release=true" in commands[0]


@pytest.mark.parametrize("flag", ["-n", "--dry-run"])
def test_forwarded_dry_run_previews_release_reuse_in_one_phase(
    benchmark: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    cache(benchmark)
    launched: list[list[str]] = []

    def launch(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        launched.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("afabench.release.config_run.subprocess.run", launch)
    command = snakemake_arguments(["--configfile", str(benchmark), "--", flag])
    assert run_config(command) == 0
    assert len(launched) == 1
    assert "release_preview=true" in launched[0]
    assert "reuse_release=true" in launched[0]
    assert "all_generate_datasets" in launched[0]


def test_output_override_selects_its_own_cache(
    benchmark: Path, tmp_path: Path
) -> None:
    cache(benchmark)
    command = arguments(benchmark, f"output_root={tmp_path / 'another'}")
    assert config_run_commands(command) == [command]


def test_dataset_target_does_not_enable_reuse(benchmark: Path) -> None:
    cache(benchmark)
    command = snakemake_arguments(
        [
            "--configfile",
            str(benchmark),
            "--target",
            "all_generate_datasets",
        ]
    )
    commands = config_run_commands(command)
    assert len(commands) == 1
    assert "reuse_release=false" in commands[0]


def test_failed_preparation_stops_before_reuse(
    benchmark: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache(benchmark)
    launched: list[list[str]] = []

    def launch(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert not check
        launched.append(command)
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr("afabench.release.config_run.subprocess.run", launch)
    assert run_config(arguments(benchmark)) == 23
    assert len(launched) == 1
    assert "all_generate_datasets" in launched[0]
