"""
No-train orchestration pipeline.

This variant skips dataset generation, classifier training, method training, and
pretraining rules. It keeps evaluation, transformation, aggregation, and
visualization rules for existing trained outputs.

Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    reference_methods (list[str], default=[]): Methods whose plotting-ready
        tables were restored from a benchmark release; aggregated with
        `methods` but never produced. See pipeline.smk.
    datasets (list[str], required): Subset of datasets to run
    dataset_realization_indices (list[int], default=[0,1,2,3,4]): Subset of
        random seeds
    device (str, default='cpu'): Deprecated invocation-wide device for
        evaluation, with a warning; cannot be combined with execution.
    execution (mapping, default={}): CPU/cuda stage defaults and method/stage
        overrides for training and evaluation. Overrides take precedence.
    execution_site_file (str, required for SLURM submission): Profile-owned
        YAML allocation map; submitting without one fails before any job.
        Alternatively provide execution_site in a configuration file. A CLI
        --config replaces the workflow profile's config, so repeat
        execution_site_file=<site>/site.yaml whenever passing --config.
    use_wandb (bool, default=True): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Initialization strategy
    eval_dataset_split (str, default='test'): Dataset split for evaluation

Execution configuration and required files:
    Methods retain their independent scripts and native bundle/result paths.
    Site profiles own CPU/GPU partition, account and GPU request syntax;
    CPU counts, memory and runtime remain separate resource settings.
    Invalid selected-method execution fails before submission.
    See docs/how-to/mixed_execution.md for the execution YAML format,
    site.yaml requirements, migration policy and captured-submission tests.

Job records:
    Evaluation and transformation jobs write a job record beside their
    output, as in pipeline.smk; see docs/reference/job_records.md.

Usage (add existing scientific --configfile inputs):
    snakemake -s extra/workflow/snakefiles/orchestration/pipeline_no_train.smk \
        --workflow-profile extra/workflow/profiles/mixed-gres \
        --configfile <scientific.yaml> <execution.yaml> -n -p all_eval_methods
    Remove -n to submit from an authorized SLURM controller with shared files.

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
      bundles, evaluation tables and time files under extra/output through
      OUTPUT_LAYOUT (afabench/core/output_layout.py), the native layout that
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
        unmaskers.yaml

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

# NOTE: exclude training rules!
# include: "../rules/training.smk"
# include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
