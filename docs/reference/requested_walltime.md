# Requested walltime

A site's `workflow/profiles/site/<site>/site.yaml` can contain a top-level
`walltime` table beside `execution_site`. The pipeline launcher selects
this file with `--site <site>`. Without an `execution_site_file`, an inline
`walltime` mapping in the Snakemake configuration is used instead.

This is scheduling configuration, not a method hyperparameter or an
acquisition budget. It sets Snakemake's `runtime` resource: the requested
wall-clock time limit of each job. It does not change the number of GPUs.

## Fields

| Location | Fields |
| --- | --- |
| `walltime` | Optional `default_hours` and `rules` mapping. |
| `walltime.rules.<rule>` | Optional `default_hours` and an identity selector mapping, where supported below. |
| `...methods.<method>` or `...pretrained_models.<model>` | Optional `default_hours` and `datasets` mapping. |
| `...datasets.<dataset key>` | Numeric hours for this rule, identity and dataset. |

Every hours value must be a positive finite YAML number (integer or float),
not a string, boolean or null. Hours are converted once to whole minutes,
rounding upward: `1.01` hours becomes 61 minutes; `0.05` becomes 3 minutes.
Unknown fields, rules, identities and dataset keys fail before submission,
even in unselected entries. Method and pretrained-model identities must be
in the checkout's catalogs under `workflow/conf/method_options/` and
`workflow/conf/pretrain_mappings/`, or in the invocation's `method_options`
and `pretrain_mapping`. Dataset keys must exist under
`conf/components/dataset_key/`. Selecting a smaller pipeline profile does
not invalidate entries for other catalog identities.

| Rule | Identity selector |
| --- | --- |
| `pretrain_model` | `pretrained_models`, using the `pretrained_model_name` wildcard, not a downstream method name. |
| `train_method`, `eval_method`, `train_classifier_for_method`, `transform_eval_data` | `methods`, using the `method` wildcard. |
| `dataset_generation`, `train_classifier`, `merge_eval_perf`, `split_by_classifier_type`, `collect_job_records`, `plot_eval_perf`, `plot_eval_actions`, `plot_time` | None; only `default_hours` is supported. |

Aggregate targets such as `all` and `all_pretrain_models` run no command
and have no walltime entry. A shared external classifier uses
`train_classifier.default_hours`, without a method identity.

## Precedence

For each job, the first matching value is used:

1. Explicit `--set-resources <rule>:runtime=<minutes>`.
2. Rule + method/pretrained model + dataset.
3. Rule + method/pretrained model `default_hours`.
4. Rule `default_hours`.
5. Site `default_hours`.
6. Snakemake's `default-resources.runtime`, or the scheduler's default when
   no runtime is configured.

Omitted entries preserve the lower-priority fallback. An absent table
leaves existing Snakemake resource behavior unchanged. Dataset realization
indices, seeds, hard budgets and soft-budget parameters do not participate
in lookup. Memory and CPU sizing remain in the site's `config.yaml`.

Snakemake treats a profile's `set-resources` exactly like command-line
resource overrides. When adding a table to a custom profile, remove its
`set-resources.*.runtime` entries so they cannot shadow the table. Shipped
profiles have already moved their rule limits to `site.yaml`, retaining
those limits. Their table is not discarded when a command-line
`--set-resources` replaces the profile's resource section.

## MiniBooNE example

These limits are **illustrative, not measured runtime recommendations**.
Choose real limits from measured job durations with a safety margin; jobs
that exceed the requested limit can be killed by Slurm.

```yaml
walltime:
  default_hours: 2
  rules:
    train_classifier:
      default_hours: 2
    pretrain_model:
      default_hours: 72
      pretrained_models:
        dime:
          default_hours: 8
          datasets: {miniboone: 4}
        l2m_real_feature_prior:
          datasets: {miniboone_subset: 6}
        l2m_synthetic_feature_prior:
          datasets: {miniboone_subset: 6}
        l2m_official:
          datasets: {miniboone_subset: 6}
    train_method:
      default_hours: 72
      methods:
        dime:
          datasets: {miniboone: 2}
        l2m_real_feature_prior:
          datasets: {miniboone_subset: 1}
        l2m_synthetic_feature_prior:
          datasets: {miniboone_subset: 1}
        l2m_official:
          datasets: {miniboone_subset: 1}
    eval_method:
      default_hours: 72
      methods:
        dime:
          datasets: {miniboone: 0.5}
        l2m_real_feature_prior:
          datasets: {miniboone_subset: 2}
        l2m_synthetic_feature_prior:
          datasets: {miniboone_subset: 2}
        l2m_official:
          datasets: {miniboone_subset: 2}

execution_site:
  # Keep the site's existing CPU/GPU allocations here.
  ...
```

## Observation and lifecycle

The [interactive preview](interactive_preview.md#panes) shows each job's
resolved `runtime` as its Time limit, and dry runs with `list_jobs=true`
show it, both in minutes.
A sparse table falling back to an input-dependent default expression may
show `<TBD>` until inputs are available. The compute planner resolves the
same resources and reports `time_limit_minutes` in its
[per-job CSV](estimate_compute_command.md#per-job-csv); job records capture
the final time limit too. A limit that cannot yet be resolved is empty in
the CSV. Compute-estimate totals still use measured job durations, not
requested time limits.

For one GPU, a 72-hour time limit requests 72 GPU-hours of potential use;
it does not mean the job will consume that much or that the account was
granted that allocation. Slurm association runtime limits can restrict
concurrency based on these commitments (`AssocGrpGRESRunMinutes`).

Editing the table affects future submissions only. It does not update jobs
already running or pending, and changing walltime alone does not rerun
completed outputs.
