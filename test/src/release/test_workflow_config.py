"""Release configuration comes from the run, not the packaging checkout."""

from pathlib import Path

import pytest
from snakemake.cli import parse_config

from afabench.release.workflow_config import (
    record_workflow_config,
    release_workflow_config,
    resolve_workflow_config,
)


@pytest.mark.parametrize(
    "value",
    [
        "True",
        "False",
        "true",
        "false",
        "1",
        "0.5",
        "[1, 2]",
        "{enabled: false, budgets: [1, 2]}",
        "null",
        "",
        "[unbalanced",
    ],
)
def test_override_assertions_use_snakemake_value_parsing(
    tmp_path: Path,
    value: str,
) -> None:
    configfile = tmp_path / "run.yaml"
    configfile.write_text("{}\n")
    arguments = [f"setting={value}"]

    record = resolve_workflow_config(
        profile=None,
        configfiles=[configfile],
        overrides=arguments,
    )

    assert record.overrides == parse_config(arguments)


def record_run(tmp_path: Path, *, methods: str = "alpha") -> Path:
    configfile = tmp_path / "run.yaml"
    configfile.write_text(f"methods: [{methods}]\n")
    root = tmp_path / "output"
    record_workflow_config(
        root,
        profile=None,
        configfiles=[configfile],
        overrides={"initializer": "warm"},
        merged={"methods": [methods], "initializer": "warm"},
    )
    return root


def test_save_time_config_changes_do_not_change_the_record(
    tmp_path: Path,
) -> None:
    root = record_run(tmp_path)
    (tmp_path / "run.yaml").unlink()

    record = release_workflow_config(root)

    assert record.merged == {"methods": ["alpha"], "initializer": "warm"}
    assert record.overrides == {"initializer": "warm"}
    assert len(record.configfiles[0].sha256) == 64


def test_supplied_matching_configuration_is_accepted(tmp_path: Path) -> None:
    root = record_run(tmp_path)

    assert release_workflow_config(
        root,
        configfiles=[tmp_path / "run.yaml"],
        overrides=["initializer=warm"],
    ) == release_workflow_config(root)


@pytest.mark.parametrize("change", ["file", "override", "profile"])
def test_supplied_conflicting_configuration_is_refused(
    tmp_path: Path, change: str
) -> None:
    root = record_run(tmp_path)
    configfile = tmp_path / "run.yaml"
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(f"configfile: {configfile}\n")
    if change == "file":
        # Even an equivalent merged config must not disguise a changed file.
        configfile.write_text("methods: [alpha] # edited after the run\n")

    with pytest.raises(ValueError, match="does not match"):
        release_workflow_config(
            root,
            profile=profile if change == "profile" else None,
            configfiles=[configfile],
            overrides=[
                "initializer=cold"
                if change == "override"
                else "initializer=warm"
            ],
        )


def test_a_later_executing_invocation_replaces_the_configuration(
    tmp_path: Path,
) -> None:
    root = record_run(tmp_path)
    record_run(tmp_path, methods="beta")

    assert release_workflow_config(root).merged["methods"] == ["beta"]


def test_missing_run_record_is_an_error_even_with_supplied_config(
    tmp_path: Path,
) -> None:
    configfile = tmp_path / "run.yaml"
    configfile.write_text("methods: [alpha]\n")

    with pytest.raises(FileNotFoundError, match="Execute pipeline"):
        release_workflow_config(tmp_path / "output", configfiles=[configfile])
