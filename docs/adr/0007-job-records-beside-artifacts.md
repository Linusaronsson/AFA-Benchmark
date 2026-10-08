---
status: accepted
---

# Job records sit beside artifacts, and compute estimates come from raw job durations

_This amends [ADR 0006](0006-release-manifest-indexes-artifact-provenance.md):
the release manifest also describes a release's job duration table, which
is built from job records rather than provenance records and ships beside
the manifest, outside `output/`._

A full benchmark run costs thousands of CPU and GPU hours, so users need a
**compute estimate** before they launch one. Today each timed job writes a
bare `*_time.txt` holding seconds only: it does not say which device or
hardware ran the job or how many CPUs and GPUs it had. Dataset generation
and classifier training are not timed at all.

We replace `*_time.txt` with a **job record**, one JSON file per pipeline job
for every computational stage. It holds the job's identity, its job
duration, and its allocation (device, CPUs, GPUs, GPU and CPU model, host,
SLURM job id). A wrapper around the stage's script writes it beside the
artifact the job produced. Failed jobs and jobs killed at the time limit
write their record to an undeclared directory instead, because Snakemake
deletes a failed job's declared outputs. An aggregation rule collects every
record into the **job duration table**, one row per job with no
aggregation, and the table is a payload category of benchmark releases. The
estimator plans jobs from the same Snakemake invocation the user would run.
It matches each planned job to measured job durations with a fixed fallback
order: an exact match, then a pool over seeds, realizations and budgets
within the same stage, name, dataset and device. It reports core-hours and
GPU-hours.

## Considered options

- **Put the timing in the provenance record (ADR-0002).** Rejected: a
  provenance record describes the artifact and should be identical however
  often the artifact is reproduced. Timing and hardware describe one run of
  the job and differ every time. A script also cannot time its own
  interpreter start-up. ADR-0002 rejects sidecars for provenance because a
  copy can separate them from their artifact. That risk is acceptable here,
  since losing a job record never makes an artifact uninterpretable.
- **A committed, hand-curated table of per-method average durations.**
  Rejected: averaging discards how job duration scales with hard budget and
  soft-budget parameter. Aggregating is the job of the estimator and of
  visualization, not of storage. Measured durations also depend on hardware,
  so they belong to the release that measured them, not to the code.
- **Keep `*_time.txt` and add a sidecar for the allocation.** Rejected: no
  benchmark release existed yet, so a clean format change cost nothing.

## Consequences

- `combined_time_results/`, the per-method time merges and `plot_time` are
  replaced by, or ported to, the job duration table.
- No durations are normalized across hardware. An estimate states which
  release and site its job durations came from.
- Durations from smoke tests are refused, because they would make every
  estimate look far too cheap.
- The release manifest (ADR 0006) gains a `job_duration_table` entry and
  payload category (manifest version 3). Unlike the rest of its index, the
  entry comes from job records, since a run's timing is no artifact's
  provenance. The table is a file beside the manifest, not in `output/`:
  the pipeline rebuilds `merged_results/job_duration_table.parquet` from
  the local output root on every `all` run and would replace a restored
  release table there. Job records do not set the release's
  `execution_mode`; the table carries its own smoke flag, so a failed smoke
  attempt left among the failed records does not block a production
  release.
- The estimator plans through Snakemake's own command-line handling, but
  Snakemake 9.12 has no public way to list a DAG's jobs with their resolved
  resources without running them. It replaces `Workflow.execute` and calls
  private members (`_prepare_dag`, `_build_dag`), so Snakemake is pinned
  below 9.13 in `pyproject.toml`. A newer version is allowed once the
  planner's workflow tests (`test/workflow/test_compute_estimate_planning.py`)
  pass on it.
- The wrapper command holds the job's allocation and is a rule param
  computed from `resources` and `threads`. Snakemake does not track such
  params for reruns, so raising a time limit or CPUs reruns no finished
  job; `test/workflow/test_job_records.py` checks that on a Snakemake
  upgrade too.
