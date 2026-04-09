# Pipeline explanation

The pipeline uses [Snakemake](https://snakemake.readthedocs.io/) for workflow orchestration and parallelization. This means dependencies are automatically tracked, and jobs are only rerun if their inputs change.

The whole pipeline is executable with the following command:
```shell
WANDB_PROJECT=afabench \
  uv run snakemake \
    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    all \
    --configfile \
      extra/workflow/conf/eval_hard_budgets.yaml \
      extra/workflow/conf/methods.yaml \
      extra/workflow/conf/method_sets.yaml \
      extra/workflow/conf/method_options.yaml \
      extra/workflow/conf/pretrain_mapping.yaml \
      extra/workflow/conf/soft_budget_params.yaml \
      extra/workflow/conf/unmaskers.yaml \
      extra/workflow/conf/classifier_names.yaml \
      extra/workflow/conf/datasets_main.yaml \
    --config \
      eval_dataset_split=val \
      "dataset_instance_indices=[0,1]" \
      smoke_test=false \
      use_wandb=true \
      device=cpu \
    --jobs 8
```

This will attempt to run 8 jobs in parallel locally on your computer, in order to produce everything that the `all` [rule](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html) requires. The `all` rule is the final target that orchestrates the entire pipeline: it generates datasets, trains classifiers, pretrains models, trains methods, evaluates them, and produces final plots. We also support [SLURM integration](slurm_integration.md).

## Configuration overview

All configuration files should be edited in place in `extra/workflow/conf/`. Below we discuss the meaning of each configuration file.

## Runtime configuration options

The `--config` section of the pipeline command allows you to customize how the pipeline runs without modifying configuration files. These options are:

### `eval_dataset_split`

Specifies which dataset split to use during evaluation.

- **Default:** `val`
- **Valid values:** `train`, `val`, `test`
- **Example:** `eval_dataset_split=test` to evaluate on the test set instead of validation set

### `dataset_instance_indices`

Specifies which random seed instances to run. This allows you to run a subset of the experiments. Each index corresponds to a different random seed for dataset generation, model initialization, and training.

- **Default:** `[0,1]`
- **Example:** `dataset_instance_indices=[0,1,2,3,4]` to run 5 different seeds
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

- **Default:** `false`
- **Example:** `use_wandb=true` to enable W&B logging
- **Requirement:** You must run `uv run wandb login` before using this option
- **Note:** Also set the `WANDB_PROJECT` environment variable (shown in the example command)

### `smoke_test`

Enables smoke testing mode, where each script runs as fast as possible while still verifying correctness. Useful for checking if the pipeline can execute successfully before running expensive experiments.

- **Default:** `false`
- **Example:** `smoke_test=true` for quick validation runs

## Datasets

The `datasets_main.yaml` file specifies which datasets are used.

## Unmaskers

`unmaskers.yaml` is a mapping from datasets to unmaskers. The values correspond to files in `extra/conf/components/unmaskers`.

For example, if `unmaskers.yaml` contains
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

`eval_hard_budgets.yaml` determines what hard budgets are used for each dataset **during evaluation**. Methods are free to use different budgets during training, see [below](#methods-and-their-soft-budget-parameters).

For example, `eval_hard_budgets.yaml` might contain
```yaml
eval_hard_budgets:
  default: [5, 10, 15]
  cube: [3, 5, 10]
  imagenette: [5, 10, 15]
```

Note that the `default` setting is used for all unlisted datasets, and that the budget describes the number of **allowed actions**, not the number of features. This is an important distinction when a different unmasker than `direct` is used.

## Methods and their soft-budget parameters

The methods require the most configuration, and use the files
- `methods.yaml`
- `method_sets.yaml`
- `pretrain_mapping.yaml`
- `method_options.yaml`
- `soft_budget_params.yaml`

`methods.yaml` is a list declaring which methods are included in the pipeline.

`method_sets.yaml` defines *method sets*, which group related methods to prevent cluttered plots when visualizing results. Each method set gets its own separate plot.

Some methods require a pretraining stage. For such methods,
`pretrain_mapping.yaml` provides the mapping to the pretraining script. For example, a `pretrain_mapping.yaml` file with contents
```yaml
pretrain_mapping:
  pvae:
    pretrain_script_name: "zannone2019"
    pretrain_params: []
```
will define a model `pvae` which is produced by the `scripts/pretrain/zannone2019.py` script. This can later be reused across different methods.

For example, `method_options.yaml` contains miscellaneous options for each method. An example configuration:
```yaml
method_options:
  ma2018_external:
    pretrained_model_name: "pvae"
    train_script_name: "ma2018_external"
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
    train_script_name: "zannone2019"
    method_specific_params:
      - "additional_generation_fraction=1.0"
    eval_batch_size:
      default: 128
    hard_budget_ignored_datasets: [imagenette]
    soft_budget_ignored_datasets: [imagenette, mnist]
```
defines two methods `ma2018_external` and `odin_model_based` which both use the same pretrained `pvae` model. Furthermore, they use different batch sizes during evaluation and ignore some datasets. `ma2018_external` is a bit special, in that it trains with a different hard budget during training compared to evaluation.

Usually during the *soft-budget* setting, the hard budget is disabled. `use_max_hard_budget_when_training_soft_budget` enforces the largest hard budget instead.

Lastly, `soft_budget_params.yaml` contains the per-dataset soft-budget parameters for each method. Each soft-budget parameter is represented as a tuple `(train_soft_budget_param, eval_soft_budget_param)`. While the `default` key **can** be used, it is recommended to tune the values for each dataset due to sensitivity issues.

## Classifiers

During evaluation, we need predictions from an *external* classifier. `classifier_names.yaml` determines which classifier is used for which dataset. You can edit these mappings to test different classifiers on your datasets. For example,
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
| `all_generate_dataset` | Generate all dataset splits (train/val/test) |
| `all_train_classifier` | Train classifiers for all datasets |
| `all_pretrain_model` | Pretrain models for all methods that require it |
| `all_train_method` | Train all methods across all datasets and budget configurations |
| `all_eval_method` | Evaluate all trained methods and produce evaluation results |

To use one of these rules, replace `all` with the desired rule name in the command above. For example, to only generate datasets and train classifiers without training methods, use:

```shell
uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk all_train_classifier ...
```
