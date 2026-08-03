from pathlib import Path

from scripts.workflow.write_run_manifest import (
    config_overrides,
    resolved_config,
)

REPO = Path(__file__).resolve().parents[2]


def test_config_overrides_extracts_only_snakemake_config() -> None:
    arguments = [
        "--rerun-incomplete",
        "--config",
        "artifact_namespace=fresh_pilot",
        "eval_dataset_split=test",
        "--printshellcmds",
    ]
    assert config_overrides(arguments) == [
        "artifact_namespace=fresh_pilot",
        "eval_dataset_split=test",
    ]


def test_resolved_config_includes_profile_and_command_line() -> None:
    config = resolved_config(
        REPO,
        "missing_data_local_nonuniform_pilot",
        ["--config", "artifact_namespace=gh200_pilot"],
    )
    assert config["artifact_namespace"] == "gh200_pilot"
    assert config["datasets"] == ["cube_nonuniform_costs"]
    assert config["dataset_instance_indices"] == [0]
