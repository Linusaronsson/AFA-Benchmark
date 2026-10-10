"""Submitting containers/build.sbatch from a site profile's allocations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "containers" / "submit_builds.py"
SITE_FILES = sorted((REPO_ROOT / "workflow/profiles/site").rglob("site.yaml"))


def submit(
    arguments: list[str], tmp_path: Path
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the script with a fake sbatch that records its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture = tmp_path / "sbatch.jsonl"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(capture)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    sbatch.chmod(0o755)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )
    calls = (
        [json.loads(line) for line in capture.read_text().splitlines()]
        if capture.exists()
        else []
    )
    return completed, calls


def write_site(tmp_path: Path, text: str) -> Path:
    site = tmp_path / "site.yaml"
    site.write_text(text)
    return site


def test_arrhenius_builds_on_each_allocation_with_its_account(
    tmp_path: Path,
) -> None:
    completed, calls = submit(["arrhenius"], tmp_path)

    assert completed.returncode == 0, completed.stderr
    containers = REPO_ROOT / "containers"
    assert calls == [
        [
            "-A",
            "naiss2026-4-1737-cpu",
            "-p",
            "cpu",
            f"--output={containers}/build-%j.log",
            "containers/build.sbatch",
            str(containers),
        ],
        [
            "-A",
            "naiss2026-4-1737-gpu",
            "-p",
            "gpu",
            "--gpus=1",
            f"--output={containers}/build-%j.log",
            "containers/build.sbatch",
            str(containers),
        ],
    ]


def test_dry_run_prints_commands_without_submitting(tmp_path: Path) -> None:
    completed, calls = submit(["arrhenius", "--dry-run"], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert calls == []
    assert completed.stdout.count("sbatch -A naiss2026-4-1737-") == 2


def test_gres_model_and_extra_arguments_reach_sbatch(tmp_path: Path) -> None:
    site = write_site(
        tmp_path,
        "execution_site:\n"
        "  cpu:\n"
        "    slurm_partition: cpu\n"
        "    slurm_account: acct\n"
        "    slurm_extra: '--qos=long'\n"
        "    image: images/afabench-x86_64.sif\n"
        "  gpu:\n"
        "    slurm_partition: gpu\n"
        "    gpu: 2\n"
        "    gpu_model: a100\n"
        "    image: images/afabench-aarch64.sif\n",
    )

    completed, calls = submit([str(site)], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert calls[0][:5] == ["-A", "acct", "-p", "cpu", "--qos=long"]
    assert calls[1][:3] == ["-p", "gpu", "--gpus=a100:2"]
    assert calls[1][-1] == str(REPO_ROOT / "images")


def test_one_image_named_by_both_allocations_is_built_once(
    tmp_path: Path,
) -> None:
    site = write_site(
        tmp_path,
        "execution_site:\n"
        "  cpu:\n"
        "    slurm_partition: cpu\n"
        "    image: containers/afabench-x86_64.sif\n"
        "  gpu:\n"
        "    slurm_partition: gpu\n"
        "    gres: gpu:T4:1\n"
        "    image: containers/afabench-x86_64.sif\n",
    )

    completed, calls = submit([str(site)], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            "execution_site:\n  cpu:\n    slurm_partition: cpu\n",
            "names no image",
        ),
        (
            "execution_site:\n  cpu:\n    image: containers/mine.sif\n",
            "not named afabench-<arch>.sif",
        ),
        ("execution_site:\n  cpu:\n    - image\n", "unsupported YAML"),
    ],
)
def test_unbuildable_site_fails_without_submitting(
    tmp_path: Path, text: str, error: str
) -> None:
    completed, calls = submit([str(write_site(tmp_path, text))], tmp_path)

    assert completed.returncode != 0
    assert error in completed.stderr
    assert calls == []


@pytest.mark.parametrize("site", SITE_FILES, ids=lambda path: path.parent.name)
def test_shipped_site_files_read_as_yaml_does(site: Path) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        from submit_builds import read_site_yaml  # noqa: PLC0415
    finally:
        sys.path.remove(str(SCRIPT.parent))

    assert read_site_yaml(site.read_text(), str(site)) == yaml.safe_load(
        site.read_text()
    )
