# L2M compute and utilization probe

Measured on Arrhenius, 2026-10-11, code revision `ff521e1a`.

## Scope

Two L2M feature sources, MiniBooNE realization 0, one NVIDIA GH200 120GB per
job, 8 requested CPU cores. Production architecture and precision, batch
size 8, sequence length 1000, 64 validation tasks, validation intervals
500 (pretraining) / 200 (training), context set size 900, evaluation batch
size 128. Only the number of fit steps was shortened to 1000 per stage.
The external classifier was fitted for one epoch to provide a valid bundle.
Evaluation used 2048 test instances per hard budget (5, 10, 15).

These are throughput probes, not benchmark-quality results. Outputs are
isolated from `output/production`; do not add shortened job durations to the
production estimator. Dataset generation and classifier training were not
extrapolated. DIME is not included.

## Measurements

Fit timings use the interval between checkpoint log entries: steps 500 to
1000 for pretraining, 200 to 1000 for training. Thus they exclude process
startup and first-step warmup, but include the production validation cadence
and checkpoint copying. GPU statistics use the same intervals, sampled once
per second (18–41 samples per interval). GPU utilization is NVML busy-time,
not a percentage of peak FLOPS or achieved occupancy.

| Feature source | Stage | Seconds/step | Full-stage GPU-hours | Mean GPU busy-time | p95 busy-time | Peak GPU memory | Mean power |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real | Pretraining | 0.041702 | 1.158 | 33.6% | 42.0% | 1922 MiB | 259 W |
| Synthetic | Pretraining | 0.036716 | 1.020 | 38.4% | 45.2% | 1922 MiB | 269 W |
| Real | Training | 0.051978 | 0.722 | 53.7% | 58.0% | 2566 MiB | 313 W |
| Synthetic | Training | 0.051018 | 0.709 | 55.1% | 59.0% | 2565 MiB | 320 W |

Full-stage hours scale to 100000 pretraining steps or 50000 training steps,
one GPU per job. Two realizations of both variants therefore require about
**7.22 GPU-hours for fitting**. Provision approximately **10 GPU-hours** for
L2M fitting and evaluation, excluding shared prerequisites, DIME, queueing
and retries. This extrapolates from one realization and short steady-state
windows, not a measured full training run.

Evaluation start-to-save times for 2048 instances:

| Feature source | Hard budget 5 | Hard budget 10 | Hard budget 15 |
| --- | ---: | ---: | ---: |
| Real | 2.676 s | 3.513 s | 4.400 s |
| Synthetic | 2.644 s | 3.482 s | 4.230 s |

The full test split has 26014 instances (confirmed by loading its bundle).
Linear extrapolation of these times adds about 0.15 GPU-hours across all
12 evaluations; allowing process startup gives roughly 0.2 GPU-hours.
Because these timings include output serialization and warmup, scaling is
approximate. Overall L2M fitting plus evaluation is approximately **7.4
GPU-hours**. With four GPUs available concurrently, the longest fit chain is
about 1.9 hours, excluding shared prerequisites and scheduling delays.

## Efficiency interpretation

The fit stages use only about 1.9–2.5 GiB of GPU memory and moderate GPU
busy-time. Host CPU use averaged roughly 1.8–2.4 core equivalents over the
whole fit processes despite requesting 8 cores. These observations suggest
host/task-generation overhead or small-kernel launch overhead, but they do
not identify the bottleneck; a CPU/CUDA profiler is needed before optimizing.

Do not silently increase training batch size or change precision: that
changes the paper recipe and may change benchmark results. A separate
follow-up experiment could profile task generation and transfers, or test
running independent fits concurrently on one GPU. There is ample memory
headroom, but throughput improvement must be measured rather than assumed.

## Runs and artifacts

- `3680167_0`, `3680167_1`: pretraining completed; training stopped because
  this experiment's command omitted required `soft_budget_param=null`.
  A local Hydra config-conversion reproducer went red without the argument
  and green with it. No library change was necessary.
- `3680251_0`, `3680251_1`: monitoring stopped before training. Physical
  `SLURM_JOB_GPUS` indices differ from NVML indices inside Arrhenius device
  isolation. The corrected script records the visible GPU UUID and verifies
  that exactly one GPU is visible. Original monitoring also showed exactly
  one visible UUID per job, so its pretraining samples remain valid.
- `3680302_0`, `3680302_1`: completed training and evaluation, reusing the
  first run's pretrained bundles. Elapsed allocations: 2m01s and 2m20s.

Local logs and GPU CSVs: `runs/3680167_{real,synthetic}/` (ignored by git).
Remote artifacts, including all bundles:
`~/projects/AFA-Benchmark/experiments/l2m_compute_probe/runs/`.
Original GPU samples are `gpu-samples.csv`; successful retry samples are
`gpu-samples-3680302.csv`. Logs and GPU timestamps use local time (UTC+2);
`stages.tsv` records UTC. UUIDs can differ between stages because retries
were scheduled on different nodes.

Reproduce from the remote checkout root after creating `runs/`:

```bash
sbatch experiments/l2m_compute_probe/run.sbatch
```

The array runs both feature sources, each capped at one GPU-hour. To reuse
completed pretraining from an earlier probe, pass
`--export=ALL,PROBE_REUSE_JOB=<original array id>` to `sbatch`.
