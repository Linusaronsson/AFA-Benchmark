"""Executable fake SLURM service: records sbatch and runs its real job wrapper."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

command = Path(sys.argv[0]).name
args = sys.argv[1:]
capture = Path(os.environ["CAPTURE"])
if command == "sbatch":
    with capture.open("a") as stream:
        stream.write(json.dumps(args) + "\n")
    job_id = len(capture.read_text().splitlines())
    wrap = next(
        arg.removeprefix("--wrap=")
        for arg in args
        if arg.startswith("--wrap=")
    )
    result = subprocess.run(
        shlex.split(wrap), capture_output=True, text=True, check=False
    )
    capture.with_name(f"job-{job_id}.log").write_text(
        result.stdout + result.stderr
    )
    status = "COMPLETED" if result.returncode == 0 else "FAILED"
    capture.with_name(f"job-{job_id}.status").write_text(status)
    print(job_id)
elif command == "sacctmgr":
    print("cpu-account\ngpu-account\nother-cpu\nother-gpu")
elif command == "sacct":
    for status_file in sorted(capture.parent.glob("job-*.status")):
        job_id = status_file.stem.removeprefix("job-")
        print(f"{job_id}|{status_file.read_text()}")
elif command == "srun":
    while args and args[0].startswith("-"):
        args.pop(0)
    sys.exit(subprocess.run(args, check=False).returncode)
