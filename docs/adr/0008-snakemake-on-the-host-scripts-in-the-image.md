---
status: accepted
---

# Snakemake runs on the host, natively per architecture; scripts run in the image

Arrhenius limits a project to 250,000 files, and an AFABench venv alone is
about 85,000, so the locked environment is packed into one Apptainer
**image** per CPU architecture (`containers/`). The login and CPU nodes are
x86_64; the GPU nodes are aarch64. We run each rule's command, job record
wrapper included, through `apptainer exec` with the image its allocation
names (`execution_site.<cpu|gpu>.image`). Snakemake runs outside the image,
on every node, from a small `orchestration` dependency group that
`containers/build.sbatch` installs per architecture beside the image. A
`python` shim, `containers/bin/python`, picks the environment of the node's
architecture. Issue #97 holds the investigation.

The reason is the SLURM executor's process chain, which a reader would not
guess from the Snakefiles. The login node's Snakemake submits each job as
`sbatch --wrap="<python> -m snakemake ... --executor slurm-jobstep"`. That
job-side Snakemake parses the Snakefile, which imports `afabench.core`, and
starts a third Snakemake with `srun -n1 <its own sys.executable> -m
snakemake`, which runs the rule's command.

## Considered options

- **Every Snakemake in the image** (a `python` shim that runs `apptainer
  exec`). Rejected: the job-side Snakemake calls `srun`, which the image
  lacks. It also hands `srun` its own `sys.executable`, a path that exists
  only inside the image.
- **Snakemake's `container:` directive.** It accepts a per-job callable, so
  an image per allocation was possible. Rejected: Snakemake still runs on the
  host, so it needs the same per-architecture environment. Its
  `--apptainer-args`, and so `--nv`, are global rather than per job.
- **One host environment with the login node's interpreter.** Rejected: the
  slurm executor starts each job's Snakemake with the login node's
  `sys.executable`, an x86_64 binary that cannot run on a GPU node.

## Consequences

- A site whose jobs run in an image drops `software-deployment` from
  `--shared-fs-usage`, so jobs start a bare `python`, and puts
  `containers/bin` first on `PATH` with `--precommand`
  (`workflow/profiles/site/arrhenius/config.yaml`).
- Sites without an image are unaffected: the image prefix is empty and the
  rendered commands are unchanged.
- An image or orchestration environment built from another `uv.lock` is
  refused: an image fails the plan before submission, and an environment is
  never found, because its directory is named after the lock's hash. Any
  `uv.lock` change, dev tools included, needs a rebuild.
- The image param takes `resources` only so that Snakemake does not track
  it: adding, changing or rebuilding an image reruns no job.
- Job records and provenance records name no image. The commit fixes the
  locked dependencies, and the host the architecture.
