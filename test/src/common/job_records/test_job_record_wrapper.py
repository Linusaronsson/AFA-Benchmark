"""
The job record wrapper run as a subprocess, as a pipeline job runs it.

Only this seam can show a time-limit kill without a real scheduler: SLURM
sends the job SIGTERM before it kills it at its time limit.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def wrapper(tmp_path: Path, *command: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "afabench.core.job_record",
        "--record",
        str(tmp_path / "output/job/model.job_record.json"),
        "--failed-record",
        str(tmp_path / "output/failed_job_records/job/model.job_record.json"),
        "--stage",
        "training",
        "--device",
        "cpu",
        "--name",
        "alpha",
        "--",
        *command,
    ]


def failed_records(tmp_path: Path) -> list[dict[str, Any]]:
    folder = tmp_path / "output/failed_job_records/job"
    return [
        json.loads(path.read_text())
        for path in sorted(folder.glob("model.*.job_record.json"))
    ]


def test_sigterm_leaves_a_timeout_record_in_the_failed_records(
    tmp_path: Path,
) -> None:
    started = tmp_path / "started"
    process = subprocess.Popen(
        wrapper(tmp_path, "sh", "-c", f"touch {started} && exec sleep 60")
    )
    deadline = time.monotonic() + 30
    while not started.exists() and time.monotonic() < deadline:
        assert process.poll() is None, "the wrapper exited early"
        time.sleep(0.05)
    time.sleep(0.2)

    process.send_signal(signal.SIGTERM)

    # 128 + SIGTERM, as a shell reports a script ended by the signal
    assert process.wait(timeout=30) == 143
    [record] = failed_records(tmp_path)
    assert record["exit_status"] == "timeout"
    assert record["stage"] == "training"
    assert record["name"] == "alpha"
    assert 0.2 <= record["job_duration_seconds"] < 30
    assert not (tmp_path / "output/job/model.job_record.json").exists()


def test_a_failing_script_s_exit_code_propagates_to_its_failed_record(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        wrapper(tmp_path, "sh", "-c", "exit 3"), check=False
    )

    assert completed.returncode == 3
    [record] = failed_records(tmp_path)
    assert record["exit_status"] == "failed"
    assert record["exit_code"] == 3
    assert not (tmp_path / "output/job/model.job_record.json").exists()


def test_a_record_names_the_slurm_job_and_cluster_it_ran_in(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "SLURM_JOB_ID": "4242",
        "SLURM_CLUSTER_NAME": "alvis",
    }

    subprocess.run(wrapper(tmp_path, "true"), env=environment, check=True)

    record = json.loads(
        (tmp_path / "output/job/model.job_record.json").read_text()
    )
    assert record["slurm_job_id"] == "4242"
    assert record["slurm_cluster"] == "alvis"
