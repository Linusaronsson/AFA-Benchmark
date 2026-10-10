"""Running jobs inside the image their site allocation names (docs/adr/0008)."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test.workflow.submission_harness import (
    REPO_ROOT,
    SITE,
    WorkflowHarness,
)

LOCK = "# the locked dependencies\n"


def with_images(workflow: WorkflowHarness, *, lock: str = LOCK) -> None:
    """Give both allocations a fake image, built from `lock`."""
    (workflow.root / "uv.lock").write_text(LOCK)
    for hardware in ["cpu", "gpu"]:
        fake_image(workflow.root / f"containers/afabench-{hardware}.sif", lock)
    workflow.config["execution_site"] = {
        hardware: {
            **allocation,
            "image": f"containers/afabench-{hardware}.sif",
        }
        for hardware, allocation in SITE.items()
    }


def fake_image(image: Path, lock: str) -> None:
    """Write an image and the lock containers/build.sbatch records beside it."""
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("an image")
    image.with_name(f"{image.name}.uv.lock").write_text(lock)


def shell_commands(output: str) -> list[str]:
    return re.findall(r"Shell command: \n\s*(.*)", output)


@pytest.mark.parametrize(
    "site",
    [
        "vera",
        "alvis",
        "examples/mixed-gres",
        "examples/mixed-gpus",
    ],
)
def test_sites_without_an_image_run_scripts_unwrapped(
    tmp_path: Path, site: str
) -> None:
    workflow = WorkflowHarness(tmp_path)

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site" / site),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = shell_commands(result.stdout)
    assert commands
    assert all(command.startswith("python ") for command in commands)


def test_an_image_wraps_the_command_and_nothing_else(tmp_path: Path) -> None:
    plain = WorkflowHarness(tmp_path / "plain")
    plain.config["execution_site"] = SITE
    imaged = WorkflowHarness(tmp_path / "imaged")
    with_images(imaged)

    plain_result = plain.run("--dry-run")
    imaged_result = imaged.run("--dry-run")

    assert imaged_result.returncode == 0, imaged_result.stderr
    prefix = re.compile(r"^apptainer exec .*? \S+\.sif ")
    imaged_commands = shell_commands(imaged_result.stdout)
    assert imaged_commands
    assert all(prefix.match(command) for command in imaged_commands)
    unwrapped = [
        prefix.sub("", command).replace(str(imaged.root), "<root>")
        for command in imaged_commands
    ]
    assert unwrapped == [
        command.replace(str(plain.root), "<root>")
        for command in shell_commands(plain_result.stdout)
    ]


def test_a_worktrees_git_directory_is_bound_too(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for args in [
        ["init", "-q"],
        [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "initial",
        ],
        ["worktree", "add", "-q", str(tmp_path / "worktree")],
    ]:
        subprocess.run(
            ["git", "-C", str(repository), *args],  # noqa: S607
            check=True,
        )
    workflow = WorkflowHarness(tmp_path / "worktree")
    with_images(workflow)

    result = workflow.run("--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    git = str((repository / ".git").resolve())
    commands = shell_commands(result.stdout)
    assert commands
    assert all(f"--bind {git} " in command for command in commands)


# Each job starts a nested Snakemake in the fake sbatch, so this takes ~10 s.
def test_arrhenius_jobs_run_snakemake_on_the_host_and_scripts_in_the_image(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}
    (tmp_path / "uv.lock").write_text(LOCK)
    shutil.copytree(REPO_ROOT / "containers/bin", tmp_path / "containers/bin")
    for arch in ["x86_64", "aarch64"]:
        fake_image(tmp_path / f"containers/afabench-{arch}.sif", LOCK)
    # The orchestration environment build.sbatch would build for this node
    lock_hash = hashlib.sha256(LOCK.encode()).hexdigest()[:16]
    host_python = (
        tmp_path
        / f"containers/orchestration-{platform.machine()}-{lock_hash}"
        / "venv/bin/python"
    )
    host_python.parent.mkdir(parents=True)
    host_python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    host_python.chmod(0o755)

    result = workflow.submit_first_wave(
        2,
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/arrhenius"),
        target="all_train_methods",
    )

    submissions = workflow.submissions()
    assert len(submissions) == 2, result.stdout
    for args in submissions:
        wrap = next(arg for arg in args if arg.startswith("--wrap="))
        # Each job's Snakemake starts from PATH, not the login node's Python.
        assert f"export PATH={tmp_path.resolve()}/containers/bin:" in wrap
        assert " && python -m snakemake " in wrap
    scripts = [
        json.loads(line)
        for line in (tmp_path / "apptainer.jsonl").read_text().splitlines()
    ]
    assert len(scripts) == 2
    checkout = str(tmp_path.resolve())
    for args in scripts:
        gpu = "scripts/train_method/alpha.py" in args
        assert ("--nv" in args) is gpu
        assert args[args.index("--bind") + 1] == checkout
        assert args[args.index("--pwd") + 1] == checkout
        assert f"PYTHONPATH={checkout}" in args
        image = image_index(args)
        assert args[image] == (
            f"{checkout}/containers/afabench-"
            + ("aarch64" if gpu else "x86_64")
            + ".sif"
        )
        # The job record wrapper runs in the image too.
        assert args[image + 1 : image + 4] == [
            "python",
            "-m",
            "afabench.core.job_record",
        ]
    devices = sorted(args["device"] for _, args in workflow.script_arguments())
    assert devices == ["cpu", "cuda"]


def image_index(args: list[str]) -> int:
    return next(i for i, arg in enumerate(args) if arg.endswith(".sif"))


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("stale", "another uv.lock"),
        ("unrecorded", "No record of the uv.lock"),
        ("missing", "does not exist"),
        ("unknown key", "Unknown execution_site.cpu keys: ['container']"),
    ],
)
def test_a_bad_image_fails_before_submission(
    tmp_path: Path, change: str, diagnostic: str
) -> None:
    workflow = WorkflowHarness(tmp_path)
    with_images(
        workflow, lock="# an older lock\n" if change == "stale" else LOCK
    )
    site = workflow.config["execution_site"]
    assert isinstance(site, dict)
    if change == "unrecorded":
        (tmp_path / "containers/afabench-cpu.sif.uv.lock").unlink()
    if change == "missing":
        (tmp_path / "containers/afabench-cpu.sif").unlink()
    if change == "unknown key":
        site["cpu"]["container"] = site["cpu"].pop("image")

    result = workflow.run("--executor", "slurm")

    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert workflow.submissions() == []
