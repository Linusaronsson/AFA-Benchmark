"""
Full orchestration pipeline: datasets, classifiers, pretraining, training,
evaluation and plots.

Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    reference_methods (list[str], default=[]): Methods from
        method_options.yaml whose plotting-ready tables were restored from a
        benchmark release (scripts/release/snapshot.py download). They join
        method sets and merge_eval_perf, but no rule produces anything for
        them, so aggregating them never schedules their training or
        evaluation; a missing reference table fails the plan with
        MissingInputException. Their tables are expected at this config's
        eval split, initializer, dataset realizations and budgets. A method
        cannot be in both lists, and method sets without any method from
        `methods` are skipped. The time plot covers `methods` only.
    datasets (list[str], required): Subset of datasets to run. Every dataset
        key needs a file conf/components/dataset_key/<key>.yaml.
    dataset_realization_indices (list[int], default=[0,1,2,3,4]): Subset of random seeds
    device (str, default='cpu'): Deprecated invocation-wide device for
        computational jobs, with a warning. Cannot be combined with execution.
    execution_file (str, default='workflow/profiles/execution/default.yaml'):
        YAML file holding the execution mapping below. Use
        workflow/profiles/execution/cpu.yaml to run every job on CPU, as
        local runs without GPUs need. Ignored when execution or the
        deprecated device is given.
    execution (mapping, default=the execution_file's mapping): CPU/cuda defaults for the pipeline
        stages classifier_training, pretraining, training and evaluation.
        methods.<name> overrides training, evaluation and method-specific
        classifier_training; pretrained_models overrides pretraining by named
        model. External classifiers use only the classifier_training default. Overrides take precedence over defaults.
        Shipped declarations: workflow/profiles/execution/{default,cpu}.yaml.
    execution_site_file (str, required for SLURM submission): Profile-owned
        YAML allocation map; submitting without one fails before any job.
        Alternatively provide execution_site in a configuration file. A CLI
        --config replaces the workflow profile's config, so repeat
        execution_site_file=<site>/site.yaml whenever passing --config.
        An allocation may name an image (execution_site.<cpu|gpu>.image,
        relative to the repository root): its jobs then run their
        commands in it through apptainer exec, with --nv on GPU
        allocations, while Snakemake stays on the host. The image must
        exist and match uv.lock, or planning fails before submission.
        See docs/how-to/slurm_integration.md and
        docs/adr/0008-snakemake-on-the-host-scripts-in-the-image.md.
    use_wandb (bool, default=True): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests, under their own
        output root
    output_root (str, default='output/production', or
        'output/smoke' with smoke_test=true): Where the run writes its
        bundles, evaluation tables, job records, merged results and plots.
        It cannot overlap the other kind of run's default root, so smoke
        outputs never satisfy a real run.
    initializer (str, default='cold'): Initialization strategy, a file in
        conf/components/initializer/
    eval_dataset_split (str, default='test'): Dataset split for evaluation

Training contract:
    The pretrain_model and train_method rules (rules/training.smk) pass every
    pretraining and training script the plain `key=value` training contract
    rendered by workflow/src/contract_arguments.py: dataset, classifier
    and pretrained-model bundle paths, save_path, method_name (the rule's
    method wildcard), initializer, unmasker, dataset_key, hard_budget,
    soft_budget_param, device, seed, use_wandb and smoke_test (pretraining
    receives no method name, no pretrained model and no budgets). Methods
    add their own arguments through method_specific_params and
    pretrain_params. See docs/adr/0001-training-contract-as-library.md.
    Every bundle a rule writes carries a provenance record in its manifest
    (docs/adr/0002-provenance-recorded-in-artifacts.md): the training
    scripts record the contract's seed, inputs and method name; the
    classifier and dataset generation scripts record theirs.

Job records:
    Every computational job (dataset generation, classifier training,
    pretraining, training, evaluation and transformation) runs its script
    through `python -m afabench.core.job_record`, rendered by
    workflow/src/job_records.py. It writes a job record, a declared
    output named after the job's artifact with `.job_record.json` in place
    of its suffix (dataset generation: dataset_generation.job_record.json in
    the dataset realization's folder): the job's identity, job duration and resolved
    allocation (device, CPUs, GPUs, time limit). A job whose script fails, or that
    receives SLURM's time-limit SIGTERM, writes its record to the same path
    under the output root's failed_job_records/ instead, one record per attempt;
    Snakemake deletes a failed job's declared outputs. The
    collect_job_records rule, part of `all`, collects every job record
    under the output root, failed attempts included, into the job duration
    table <output_root>/merged_results/job_duration_table.parquet, one row
    per record (afabench.core.job_duration_table). The plot_time rule plots
    from it the job durations of the completed pretraining, method-specific
    classifier training, training, evaluation and transformation jobs of
    `methods` under this initializer and eval split; failed and timed-out
    attempts are left out. See docs/reference/job_records.md and
    docs/adr/0007-job-records-beside-artifacts.md.

Execution configuration and required files:
    Methods retain their independent scripts and native bundle/result paths.
    Site profiles own CPU/GPU partition, account and GPU request syntax;
    CPU counts, memory and runtime remain separate resource settings.
    Invalid selected-method and prerequisite execution fails before submission,
    including conflicting device arguments in classifier/pretraining params,
    unknown pretrained_models names and default-resources slurm_extra.
    See docs/how-to/mixed_execution.md and prerequisite_execution.md for
    execution YAML, site.yaml, migration and captured-submission tests.

Usage:
    Full benchmark, one invocation from an authorized SLURM submit host of a
    single cluster, with the repository, environment and outputs on a shared
    filesystem (execution/default.yaml places the GPU jobs):
        snakemake --profile workflow/profiles/pipeline/kdd26 \
            --workflow-profile workflow/profiles/site/<site> -n -p all
    Inspect the planned resources and device arguments, then remove -n -p.
    Full method set: use --profile workflow/profiles/pipeline/all instead.
    Local CPU smoke test without SLURM or GPUs:
        snakemake --profile workflow/profiles/pipeline/all all --jobs 8 \
            --config "datasets=[cube]" "dataset_realization_indices=[0]" \
            smoke_test=true use_wandb=false \
            execution_file=workflow/profiles/execution/cpu.yaml
    It writes under output/smoke, so a later real run still runs every
    job.
    See docs/how-to/run_the_pipeline.md and slurm_integration.md.
    Add a method to published baselines: download the baselines'
    transformed tables and the shared prerequisites into
    output/production, then run only the new method's missing work
    and the comparison plots:
        snakemake --profile workflow/profiles/pipeline/all all --jobs 8 \
            --config "methods=[my_method]" \
                "reference_methods=[random_dummy, gdfs]" \
                "datasets=[cube]" "dataset_realization_indices=[0]" \
                execution_file=workflow/profiles/execution/cpu.yaml
    See docs/how-to/compare_your_method_with_published_results.md.

CPU-only processing:
    Dataset generation (full pipeline only), transformations, aggregation and
    visualization always resolve to CPU, including with legacy device=cuda.
    These fixed stages have no execution defaults/overrides. The profile's
    execution_site.cpu allocation maps their partition/account and clears GPU
    requests; CPU counts, memory and runtime remain independently configurable.
    Conflicting rule allocation overrides fail before any submission. Heavy
    processing is submitted normally, not designated as login-node/local work.
    See docs/how-to/cpu_processing_execution.md for site requirements and
    final-target command-boundary verification.

Output namespacing:
    - Every rule addresses dataset, classifier, pretrained-model and method
      bundles, evaluation tables and job records under the output root
      (output_root) through OUTPUT_LAYOUT (afabench/core/output_layout.py), the native layout that
      release manifests and restored benchmark releases also use.
    - All initializer-dependent artifacts are stored under
      `initializer-<initializer>` to allow side-by-side comparisons
      (for example `cold` vs `missingness`) without overwriting.

Config files (--configfile):
    The merged config is validated by load_config
    (afabench/core/workflow_settings.py) through ../settings.smk, which
    defines the globals the rules read. Unknown method_options keys, a
    missing train_script_name, a pretrained_model_name outside
    pretrain_mapping, a method missing from method_options or
    soft_budget_params, and an ignored dataset that is not a dataset key all
    fail the parse, naming the method, as does a malformed pretrain_mapping
    entry, naming the pretrained model; see
    docs/reference/pipeline_configuration.md.
    Fixed definitions:
        method_options.yaml,
        pretrain_mapping.yaml,
        methods.yaml
        classifier_names.yaml
    Runtime params:
        eval_hard_budgets.yaml,
        soft_budget_params_*.yaml,
        unmaskers.yaml (values are files in conf/components/unmasker/)

    Note: method_options.yaml can include eval_to_train_hard_budget_mapping to
    specify different budgets for training vs evaluation per method/dataset.

    Note: method_options.yaml can include eval_batch_size to specify different
    batch sizes for evaluation per method and dataset. Format:
        eval_batch_size:
          default: <batch_size>
          <dataset_name>: <batch_size>

    Note: method_options.yaml can include hard_budget_ignored_datasets to skip
    hard budget training/evaluation for specific datasets per method. Format:
        hard_budget_ignored_datasets: [dataset1, dataset2, ...]
    When set, hard budget combinations are excluded for those datasets.

    Note: method_options.yaml can include soft_budget_ignored_datasets to skip
    soft budget training/evaluation for specific datasets per method. Format:
     soft_budget_ignored_datasets: [dataset1, dataset2, ...]
     When set, soft budget combinations are excluded for those datasets.

     Note: Pretraining is skipped per pretrained model for datasets ignored by
     BOTH hard and soft budgets across all methods that consume that model.
"""

include: "../settings.smk"

include: "../rules/dataset_generation.smk"
include: "../rules/training.smk"
include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
