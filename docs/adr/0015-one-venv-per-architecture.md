---
status: accepted
---

# One venv per architecture on a cluster of mixed architectures

Arrhenius's login and CPU nodes are x86_64 and its GPU nodes aarch64, so one
venv cannot serve every job: `uv.lock` pins different torch wheels for each,
and a binary of one architecture does not run on the other. Each allocation
of the site file names the venv of its nodes' architecture
(`execution_site.<cpu|gpu>.venv`, `.venv-<arch>` in the checkout), which
`workflow/venvs/build.sbatch` syncs from `uv.lock` on a node of that
architecture. This supersedes
[ADR 0008](0008-snakemake-on-the-host-scripts-in-the-image.md), which
packed the environment into an Apptainer image per architecture only
because Arrhenius then limited a project to 250,000 files, about three
venvs. The limit is now 1,000,000.

What picks the venv is the SLURM executor's process chain, which a reader
would not guess from the Snakefiles. The login node's Snakemake submits
each job as `sbatch --wrap="<python> -m snakemake ... --executor
slurm-jobstep"`, with its own `sys.executable` as `<python>` unless
`software-deployment` is dropped from `--shared-fs-usage`. That job-side
Snakemake parses the Snakefile and starts a third with `srun -n1 <its own
sys.executable> -m snakemake`, which runs the rule's command, whose `python`
comes from `PATH`.

So the Arrhenius site profile drops `software-deployment`, and its
`--precommand` puts `workflow/venvs/bin` first on `PATH`. Its `python` runs
`.venv-$(uname -m)/bin/python`, deciding on the node. The pipeline launcher
starts the submitting Snakemake through it too. Non-executing modes (dry
runs, compute estimates and interactive previews) instead use the checkout
environment. Planning retains the site's allocations but does not certify
execution readiness; requiring compute-node builds would prevent inspecting
a plan before those builds finish.

## Considered options

- **`.venv-$(uname -m)/bin` on `PATH` in the precommand itself**, without a
  shim. Rejected: the executor runs `sbatch` through a shell on the login
  node, with the precommand inside double quotes, so `$(uname -m)` would be
  expanded there and every job would get the x86_64 venv. Escaping it works
  only for as long as the executor quotes that way.
- **Every job syncs its venv first** (`uv sync --frozen` in the
  precommand). No build step and never stale, but hundreds of jobs would
  sync at once on the shared filesystem, and a `uv.lock` change during a
  run would change the venv under running jobs.
- **Keep the images** now that the limit allows venvs. Rejected: they need
  Apptainer, a second environment for Snakemake on the host, and a bind
  for every path a job reads, all for a limit that is gone.

## Consequences

- The launcher refuses execution while a venv the site names is missing or
  was built from another `uv.lock`; non-executing modes only warn.
  `build.sbatch` records the lock it synced
  from as `<venv>/afabench.uv.lock`, only once the sync succeeded, because
  the login node cannot run an aarch64 venv to ask it. Any `uv.lock` change,
  dev tools included, needs a rebuild before execution.
- A venv is synced in place, so it is rebuilt while no job of the pipeline
  runs.
- The venvs hold the project without the dev group. Snakemake is a main
  dependency, since every job's Snakemake starts from its venv.
- Sites whose nodes share one architecture name no venv, and run as before:
  the launcher starts `uv run snakemake` and jobs start the login node's
  interpreter.
- The launcher itself and `workflow/venvs/submit_builds.py` run in the
  checkout's `.venv` through `uv run`, not in these venvs: the builds are
  submitted before any of them exists, and syncing the dev group into one
  would change it under running jobs.
