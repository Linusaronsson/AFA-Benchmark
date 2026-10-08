# Job records

A [job record](../../CONTEXT.md) is a JSON file that one computational
pipeline job writes beside the artifact it produced. It holds the job's
identity, its job duration and the allocation it ran with
([ADR 0006](../adr/0006-job-records-beside-artifacts.md)). It describes the
run, not the artifact, so it is not part of the artifact's provenance record.
The schema is `afabench.core.job_record.JobRecord`, version 1.

## Where records are

Each record is a declared output of its rule, named after the artifact with
`.job_record.json` in place of the artifact's suffix. Paths are relative to
the output root `extra/output/`; `<tag>` is `initializer-<initializer>`,
`<training>` the training run's folder
(`<method>/dataset-<key>+realization_index-<k>/<pretrain folder>/train_seed-<s>+train_hard_budget-<b>+train_soft_budget_param-<p>`)
and `<evaluation>` the evaluation's subfolder
(`eval_seed-<s>+eval_hard_budget-<b>+eval_soft_budget_param-<p>`).

| Pipeline stage | Rule | Record |
| --- | --- | --- |
| `dataset_generation` | `dataset_generation` | `datasets/<key>.job_record.json`, beside the folder of all dataset realizations of `<key>` |
| `classifier_training` | `train_classifier`, `train_classifier_for_method` | `trained_classifiers/<tag>/[method-<method>+]dataset-<key>+realization_index-<k>.job_record.json` |
| `pretraining` | `pretrain_model` | `pretrained_models/<tag>/<pretrained model>/dataset-<key>+realization_index-<k>/pretrain_seed-<s>/model.job_record.json` |
| `training` | `train_method` | `trained_methods/<tag>/<training>/method.job_record.json` |
| `evaluation` | `eval_method` | `eval_results/eval_split-<split>/<tag>/<training>/<evaluation>/eval_data.job_record.json` |
| `transformation` | `transform_eval_data` | `eval_results_transformed/eval_split-<split>/<tag>/<training>/<evaluation>/eval_data.job_record.json` |

No job depends on a record, so a missing record never makes Snakemake rerun
a job whose artifact exists, for example a downloaded shared prerequisite.
Pretraining, training and evaluation also still write `*_time.txt`, which
the time aggregation reads.

## How a record is written

A rule runs its script through the wrapper command, which runs the script,
times it and writes the record:

```shell
python -m afabench.core.job_record --record <path> --stage <stage> \
    --device <cpu|cuda> [--cpus <n>] [--gpus <n>] [--smoke-test] \
    [identity options] [--time-file <path>] -- <script command>
```

Each identity field below has an option of the same name with `-` for `_`
(`--dataset-key`); an omitted option records `null`. `--time-file` also
writes the job duration in seconds there. The wrapper exits with the
script's exit code. `extra/workflow/src/job_records.py` renders the options
from the job's wildcards and its final Snakemake resources, after profile
defaults and `--set-resources` overrides.

## Fields

`null` means unknown, or not part of this job's identity, never a default.
Records are flat, so that one record is one row of a table.

| Field | Meaning |
| --- | --- |
| `job_record_version` | Schema version, 1. |

### Identity

| Field | Meaning |
| --- | --- |
| `stage` | Pipeline stage: `dataset_generation`, `classifier_training`, `pretraining`, `training`, `evaluation` or `transformation`. |
| `name` | Method name for training, evaluation and transformation; pretrained-model name for pretraining; the classifier's script name for classifier training; null for dataset generation. |
| `dataset_key` | Dataset key. |
| `dataset_realization_index` | Dataset realization index; null for dataset generation, whose one job generates every dataset realization of the key. |
| `pretrain_seed` | Seed of the pretraining run the job is part of or depends on; null without pretraining. |
| `train_seed` | Seed of the training run; for classifier training, the classifier's seed, which is its dataset realization index. |
| `eval_seed` | Seed of the evaluation. |
| `train_hard_budget`, `eval_hard_budget` | Hard budget (integer) of training and of evaluation; null in the soft-budget setting. |
| `train_soft_budget_param`, `eval_soft_budget_param` | Soft-budget parameter (number) of training and of evaluation; null in the hard-budget setting. |
| `eval_batch_size` | Evaluation batch size; evaluation only. |

### Timing

| Field | Meaning |
| --- | --- |
| `started_at`, `ended_at` | UTC ISO-8601 times the script started and ended. |
| `job_duration_seconds` | Job duration: the script's wall-clock time in seconds, measured with a monotonic clock. |
| `exit_status` | `completed` if the script exited with 0, `failed` otherwise. Snakemake deletes a failed job's declared outputs, so only `completed` records remain beside artifacts. |
| `exit_code` | The script's exit code; negative if a signal ended it. |

### Allocation

| Field | Meaning |
| --- | --- |
| `device` | `cpu` or `cuda`, the device the pipeline resolved for the job. |
| `cpus` | CPUs requested: the `cpus_per_task` resource, or the job's threads without one, as the SLURM executor requests them; null if `cpus_per_task` is negative, which leaves it to the cluster. |
| `gpus` | GPUs requested: the `gpu` resource, or the count of a `gpu[:<model>]:<n>` `gres`; 0 for none. |
| `gpu_model` | Names of the GPUs `nvidia-smi` lists in the job, comma-separated; null for jobs without GPUs or where it cannot be queried. |
| `cpu_model` | The CPU's model name from `/proc/cpuinfo`, or the platform's processor name. |
| `host` | Host name the job ran on. |
| `slurm_job_id` | `SLURM_JOB_ID` of the job; null outside SLURM. |

### Code and mode

| Field | Meaning |
| --- | --- |
| `code_commit` | Commit of the afabench checkout that ran the job; null outside a git work tree. |
| `smoke_test` | Whether the pipeline ran with `smoke_test=true`. |
