"""
Executable fake SLURM service: records sbatch and runs its real job wrapper.

It also fakes `apptainer exec`: the command runs on the host, in the given
working directory and environment, whatever the image.
"""

import json
import os
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
    # sbatch runs the wrapped command in sh, as a site's --precommand needs.
    result = subprocess.run(
        ["sh", "-c", wrap],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    capture.with_name(f"job-{job_id}.log").write_text(
        result.stdout + result.stderr
    )
    status = "COMPLETED" if result.returncode == 0 else "FAILED"
    capture.with_name(f"job-{job_id}.status").write_text(status)
    print(job_id)
elif command == "sacctmgr":
    # The last two are Arrhenius's, whose shipped site profile is run too.
    print(
        "cpu-account\ngpu-account\nother-cpu\nother-gpu\n"
        "naiss2026-4-1737-cpu\nnaiss2026-4-1737-gpu"
    )
elif command == "sinfo":
    # The cluster's default partition is marked with an asterisk.
    print("PARTITION\ncpu-queue\ngpu-queue\ncpu\ngpu\ngeneral*")
elif command == "sacct":
    for status_file in sorted(capture.parent.glob("job-*.status")):
        job_id = status_file.stem.removeprefix("job-")
        print(f"{job_id}|{status_file.read_text()}")
elif command == "apptainer":
    with capture.with_name("apptainer.jsonl").open("a") as stream:
        stream.write(json.dumps(args) + "\n")
    assert args.pop(0) == "exec"
    options: dict[str, list[str]] = {}
    while args[0].startswith("--"):
        option = args.pop(0)
        values = options.setdefault(option, [])
        if option != "--nv":
            values.append(args.pop(0))
    args.pop(0)  # The image
    environment = dict(os.environ)
    for assignment in options.get("--env", []):
        name, value = assignment.split("=", 1)
        environment[name] = value
    sys.exit(
        subprocess.run(
            args,
            cwd=options["--pwd"][0],
            env=environment,
            check=False,
        ).returncode
    )
elif command == "srun":
    while args and args[0].startswith("-"):
        args.pop(0)
    sys.exit(subprocess.run(args, check=False).returncode)
