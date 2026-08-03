"""Write immutable execution provenance before a missing-data workflow run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf


def config_overrides(arguments: list[str]) -> list[str]:
    """Extract Snakemake ``--config key=value`` dotlist entries."""
    overrides: list[str] = []
    collecting = False
    for argument in arguments:
        if argument == "--config":
            collecting = True
            continue
        if collecting and argument.startswith("-"):
            collecting = False
        if collecting and "=" in argument:
            overrides.append(argument)
    return overrides


def resolved_config(
    repo: Path, profile: str, arguments: list[str]
) -> dict[str, Any]:
    """Resolve the profile's configfiles and command-line config overrides."""
    profile_path = (
        repo / "extra/workflow/profiles/config" / profile / "config.yaml"
    )
    profile_config = cast("DictConfig", OmegaConf.load(profile_path))
    merged = OmegaConf.create()
    for configfile in profile_config.get("configfile", []):
        merged = OmegaConf.merge(
            merged, OmegaConf.load(repo / str(configfile))
        )
    merged = OmegaConf.merge(merged, profile_config.get("config", {}))
    merged = OmegaConf.merge(
        merged, OmegaConf.from_dotlist(config_overrides(arguments))
    )
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        message = "resolved workflow configuration is not a mapping"
        raise TypeError(message)
    return cast("dict[str, Any]", container)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],  # noqa: S607 - repository tool, not user input
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--mem-mb", type=int, required=True)
    parser.add_argument("--gpu-workers", type=int, required=True)
    parser.add_argument(
        "--snakemake-args", nargs=argparse.REMAINDER, default=[]
    )
    arguments = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    config = resolved_config(repo, arguments.profile, arguments.snakemake_args)
    namespace = str(config["artifact_namespace"])
    cuda_devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    git_commit = git(repo, "rev-parse", "HEAD")
    git_dirty = bool(git(repo, "status", "--short"))
    if os.environ.get("SLURM_JOB_ID") and git_dirty:
        message = "refusing a cluster run from a dirty Git checkout"
        raise RuntimeError(message)
    manifest = {
        "schema_version": 1,
        "run_id": arguments.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_namespace": namespace,
        "profile": arguments.profile,
        "eval_dataset_split": config.get("eval_dataset_split"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "command": {
            "device": arguments.device,
            "cores": arguments.cores,
            "mem_mb": arguments.mem_mb,
            "gpu_workers": arguments.gpu_workers,
            "snakemake_args": arguments.snakemake_args,
        },
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "cuda_devices": cuda_devices,
        "gpu_telemetry_path": (
            f"extra/output/missing_data/gpu_telemetry/{namespace}/"
            f"{arguments.run_id}.csv"
            if arguments.device.startswith("cuda")
            else None
        ),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_GPUS",
                "SNIC_TMP",
            )
        },
    }
    output = (
        repo
        / "extra/output/missing_data/run_manifests"
        / namespace
        / f"{arguments.run_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"wrote run manifest {output.relative_to(repo)}")


if __name__ == "__main__":
    main()
