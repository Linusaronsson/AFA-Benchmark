# Pipeline explanation

The pipeline uses [Snakemake](https://snakemake.readthedocs.io/) for workflow orchestration and parallelization. This means dependencies are automatically tracked, and jobs are only rerun if their inputs change.

The whole pipeline is executable with the following command:
```shell
WANDB_PROJECT=afabench \
  uv run snakemake \
    --profile extra/workflow/profiles/config/all \
    all \
    --jobs 8
```

This will attempt to run 8 jobs in parallel locally on your computer, in order to produce everything that the `all` [rule](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html) requires. The `all` rule is the final target that orchestrates the entire pipeline: it generates datasets, trains classifiers, pretrains models, trains methods, evaluates them, and produces final plots. We also support [SLURM integration](slurm_integration.md).

## Configuration overview

Configuration files are organized into subdirectories under
`extra/workflow/conf/`. Each subdirectory contains multiple named variants
(e.g., `all.yaml`, `kdd26.yaml`). The command above uses the
`extra/workflow/profiles/config/all` profile, which bundles the commonly used
`all.yaml` config files and the pipeline Snakefile. Use
`extra/workflow/profiles/config/cpu_methods`,
`extra/workflow/profiles/config/gpu_methods`, or
`extra/workflow/profiles/config/kdd26` when you want those preset config
combinations instead. Below we discuss the meaning of each configuration group.

## Runtime configuration options

The `--config` section of the pipeline command allows you to customize how the pipeline runs without modifying configuration files. These options are:

### `eval_dataset_split`

Specifies which dataset split to use during evaluation.

- **Default:** `test`
- **Valid values:** `train`, `val`, `test`
- **Example:** `eval_dataset_split=val` to evaluate on the validation set instead of the test set

### `dataset_instance_indices`

Specifies which random seed instances to run. This allows you to run a subset of the experiments. Each index corresponds to a different random seed for dataset generation, model initialization, and training.

- **Default:** `[0,1,2,3,4]`
- **Example:** `dataset_instance_indices=[0,1]` to run two different seeds
- **Use case:** Use fewer instances for faster debugging, more instances for more robust results

### `device`

Specifies the compute device to use for training and evaluation.

- **Default:** `cpu`
- **Valid values:**
  - `cpu` - Use CPU only
  - `cuda` - Use CUDA GPU (defaults to the first available GPU)
  - `cuda:0`, `cuda:1`, etc. - Use a specific CUDA GPU device
- **Example:** `device=cuda` to accelerate training with GPU

### `--jobs` (Snakemake parameter)

Controls the number of jobs Snakemake runs in parallel. This is not a `--config` option but a direct Snakemake flag (specified with `-j`).

- **Default:** 1 (serial execution)
- **Example:** `--jobs 8` to run 8 jobs in parallel
- **Note:** Set this to the number of CPU cores available on your machine for optimal parallelization. Be mindful of memory usage when increasing this value.

### `use_wandb`

Enables or disables [Weights & Biases](https://wandb.ai/) integration for logging metrics.

- **Default:** `true`
- **Example:** `use_wandb=false` to disable W&B logging
- **Requirement:** You must run `uv run wandb login` before using W&B
- **Note:** Also set the `WANDB_PROJECT` environment variable (shown in the example command)

### `smoke_test`

Enables smoke testing mode, where each script runs as fast as possible while still verifying correctness. Useful for checking if the pipeline can execute successfully before running expensive experiments.

- **Default:** `false`
- **Example:** `smoke_test=true` for quick validation runs

## Datasets

The `extra/workflow/conf/datasets/` directory contains dataset configuration files. Each file specifies which datasets are used in the pipeline.

## Unmaskers

`extra/workflow/conf/unmaskers/` contains files that map datasets to unmaskers. The values correspond to files in `extra/conf/components/unmaskers`.

For example, if `extra/workflow/conf/unmaskers/all.yaml` contains
```yaml
unmaskers:
  default: direct
  imagenette: 224x224_to_14x14
```
then `imagenette` will use a patch-based unmasker while all other datasets will have the "normal" unmasker that maps actions directly to features.

`extra/conf/components/unmaskers/224x224_to_14x14.yaml` contains the details about this specific unmasker:
```yaml
class_name: "ImagePatchUnmasker"
kwargs:
  image_side_length: 224  # Imagenette size
  n_channels: 3
  patch_size: 16  # 14x14 grid = 196 patches
```

## Hard budgets

`extra/workflow/conf/eval_hard_budgets/` determines what hard budgets are used for each dataset **during evaluation**. Methods are free to use different budgets during training, see [below](#methods-and-their-soft-budget-parameters).

For example, a file in `extra/workflow/conf/eval_hard_budgets/` might contain
```yaml
eval_hard_budgets:
  default: [5, 10, 15]
  cube: [3, 5, 10]
  imagenette: [5, 10, 15]
```

Note that the `default` setting is used for all unlisted datasets, and that the budget describes the number of **allowed actions**, not the number of features. This is an important distinction when a different unmasker than `direct` is used.

## Methods and their soft-budget parameters

The methods require the most configuration, and use the directories
- `extra/workflow/conf/methods/`
- `extra/workflow/conf/method_sets/`
- `extra/workflow/conf/pretrain_mappings/`
- `extra/workflow/conf/method_options/`
- `extra/workflow/conf/soft_budget_params/`

`methods/` contains files listing which methods are included in the pipeline.

`method_sets/` contains files that define *method sets*, which group related methods to prevent cluttered plots when visualizing results. Each method set gets its own separate plot.

Some methods require a pretraining stage. For such methods,
`pretrain_mappings/` provides the mapping to the pretraining script. For example, a file in `extra/workflow/conf/pretrain_mappings/` with contents
```yaml
pretrain_mapping:
  pvae:
    pretrain_script_name: "odin"
    pretrain_params: []
```
will define a model `pvae` which is produced by the `scripts/pretrain_model/odin.py` script. This can later be reused across different methods.

For example, a file in `extra/workflow/conf/method_options/` contains miscellaneous options for each method. An example configuration:
```yaml
method_options:
  eddi_external:
    pretrained_model_name: "pvae"
    train_script_name: "eddi_external"
    use_max_hard_budget_when_training_soft_budget: true
    eval_batch_size:
      default: 8
    eval_to_train_hard_budget_mapping:
      cube_nonuniform_costs:
        2: 20
        4: 20
        7: 20
    hard_budget_ignored_datasets: [mnist, fashion_mnist, imagenette]
    soft_budget_ignored_datasets: [mnist, fashion_mnist, imagenette]
  odin_model_based:
    pretrained_model_name: "pvae"
    train_script_name: "odin"
    method_specific_params:
      - "additional_generation_fraction=1.0"
    eval_batch_size:
      default: 128
    hard_budget_ignored_datasets: [imagenette]
    soft_budget_ignored_datasets: [imagenette, mnist]
```
defines two methods `eddi_external` and `odin_model_based` which both use the same pretrained `pvae` model. Furthermore, they use different batch sizes during evaluation and ignore some datasets. `eddi_external` is a bit special, in that it trains with a different hard budget during training compared to evaluation.

Usually during the *soft-budget* setting, the hard budget is disabled. `use_max_hard_budget_when_training_soft_budget` enforces the largest hard budget instead.

Lastly, files in `extra/workflow/conf/soft_budget_params/` contain the per-dataset soft-budget parameters for each method. Each soft-budget parameter is represented as a tuple `(train_soft_budget_param, eval_soft_budget_param)`. While the `default` key **can** be used, it is recommended to tune the values for each dataset due to sensitivity issues.

## Classifiers

During evaluation, we need predictions from an *external* classifier. Files in `extra/workflow/conf/classifier_names/` determine which classifier is used for which dataset. You can edit these mappings to test different classifiers on your datasets. For example,
```yaml
classifier_names:
  default: "masked_mlp_classifier"
  imagenette: "masked_vit_classifier"
```
will use a vision transformer for the `imagenette` dataset but a normal `MLP` classifier for the other datasets.

## Running specific steps of the pipeline

The whole point of Snakemake is to run jobs maximally parallelized. Still, there might be good reasons for only running specific steps across all methods. For example, perhaps all methods need to be pretrained and trained without being evaluated. For this, we provide various `all_X` rules in `extra/workflow/snakefiles/rules/helpers.smk`. These rules should replace the `all` rule from the command at the beginning of this document.

The currently available `all_X` rules are:

| Rule | Purpose |
|------|---------|
| `all_generate_datasets` | Generate all dataset splits (train/val/test) |
| `all_train_classifiers` | Train classifiers for all datasets |
| `all_pretrain_models` | Pretrain models for all methods that require it |
| `all_train_methods` | Train all methods across all datasets and budget configurations |
| `all_eval_methods` | Evaluate all trained methods and produce evaluation results |

To use one of these rules, replace `all` with the desired rule name in the command above. For example, to only generate datasets and train classifiers without training methods, use:

```shell
uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk all_train_classifiers ...
```
