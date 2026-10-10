---
status: accepted
---

# A pipeline launcher owns the command line

The pipeline is run with `workflow/run.py`, the **pipeline launcher**, rather
than with `snakemake` directly. A run is three choices: a pipeline profile
(what runs), a site profile (where it runs) and pipeline profile overrides.
The launcher takes exactly those and builds the Snakemake command, adding
what follows from them and that users should not need to know. Issue #98
holds the discussion.

Two things follow from the site profile:

- **Its allocations.** A site profile points at its allocations with
  `config: execution_site_file: <site>/site.yaml`. Snakemake treats a
  profile's `config` as the default of `--config` and replaces it, rather
  than merging, whenever `--config` is given. So any override silently
  dropped the site's allocations, and submission failed until the user
  repeated `execution_site_file=` on the command line. Every documented
  cluster command carried that line.
- **Its interpreter.** On a site whose jobs run in an image, Snakemake runs
  from the host's orchestration environment, not uv
  ([ADR 0008](0008-snakemake-on-the-host-scripts-in-the-image.md)), and
  users had to remember to start `containers/snakemake.sh` instead.

The launcher adds `execution_site_file` to the one `--config` it builds
from the overrides, and chooses the interpreter from whether the site file
names an image.

## Considered options

Every option has to reach the jobs, not just the submitting Snakemake: each
job parses the Snakefile again and needs its allocation, for its image and
the allocation checks. Snakemake 9 passes jobs the `--config`,
`--configfiles` and resource arguments and the environment, but never the
profiles, and keeps no record of `--workflow-profile` a Snakefile can read.

- **The site file path as a default resource** of the site profile, which
  jobs receive through `--default-resources`. The smallest change, and plain
  `snakemake` would have kept working. Rejected: a resource that is no
  scheduler resource, dropped again by a `--default-resources` on the
  command line, and it leaves the interpreter to the user.
- **The allocations themselves as prefixed default resources**
  (`gpu_slurm_partition`, ...), without `site.yaml`. Rejected: about ten
  fake resources on every job, and the nested CPU/GPU allocation shape the
  validation and docs use is lost.
- **A `configfile` in the site profile.** Rejected: Snakemake replaces the
  pipeline profile's `configfile` list with it, as it does `config`.
- **The Snakefile reads `--workflow-profile` from `sys.argv`** and hands
  the site file to jobs in an environment variable. Rejected: relies on
  undocumented process details on both sides.

The launcher was chosen because it puts the whole interface under the
project's control: the command a user types is the three choices, and
whatever else Snakemake needs is the launcher's concern.

## Consequences

- Plain `snakemake` still works, and still drops the site's allocations
  with `--config`; the documentation teaches only the launcher, and the
  error on cluster submission without allocations points to it.
- The launcher runs on the host's `python3` with the standard library only
  (3.9 on Arrhenius), since it runs before any environment exists; it uses
  argparse, not typer.
- `just estimate-compute` takes the launcher's arguments, so a run is
  estimated with the same arguments it is run with.
