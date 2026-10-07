# Adding a new method

An AFA method (see [`CONTEXT.md`](../../CONTEXT.md)) is added to the
benchmark as one or two plain scripts under `scripts/train_method/` and
`scripts/pretrain_model/`, registered with the pipeline. There is exactly one
hard rule for those scripts. Everything else — Hydra, the config
dataclasses, the shared helpers — is there to save you work, not to
constrain you.

## 1. The training contract

The pipeline (Snakemake) invokes your training script as a subprocess with a
fixed set of `key=value` command-line arguments, and your pretraining script
(if you have one) with its own set. These are the **training contract** and
the **pretraining contract** (glossary entries in [`CONTEXT.md`](../../CONTEXT.md)), and its design is
recorded in
[ADR-0001](../adr/0001-training-contract-as-library.md).

**The hard rule: accept the training contract's arguments on the command
line, and write a loadable bundle to the `save_path` argument.** Nothing
else about how your method trains, configures itself, or uses the dataset
key is prescribed.

The contract's field names and types are not restated here; the source of
truth is the `PretrainingContract` / `TrainingContract` dataclasses in
`afabench/fit/contract.py`. Both extend `BaseContract` and are
independent of each other: `TrainingContract` has the pretrained-model path
and the budgets, which `PretrainingContract` doesn't, but the two stages are
free to diverge further. Read that file before writing a new method's config.

## 2. Minimal method

The smallest way to satisfy the contract is a training script with no
pretraining stage that inherits `TrainingContract` for its config and uses
the optional library helpers. The ported dummy methods
(`afabench/components/methods/dummy/`) are the worked example; this section
walks through `random_dummy`.

The config dataclass adds no fields of its own:

```python
# afabench/components/methods/dummy/config.py
from dataclasses import dataclass

from afabench.fit.contract import TrainingContract, store_contract_config


@dataclass(frozen=True, kw_only=True)
class RandomDummyTrainConfig(TrainingContract):
    """The random dummy method has no hyperparameters beyond the contract."""


store_contract_config(
    name="train_random_dummy", config_class=RandomDummyTrainConfig
)
```

`store_contract_config` registers the dataclass as a Hydra structured config
under the given name; the next section covers how the YAML config pulls it
in. The training logic is a plain function that takes the resolved config and
a `FitInputs` and returns the trained method — it does not seed, log, or
save anything itself:

```python
# afabench/components/methods/dummy/train.py (abridged)
def train_random_dummy(
    contract: RandomDummyTrainConfig, inputs: FitInputs
) -> RandomWithoutClassifierAFAMethod:
    train_dataset = inputs.train_dataset()
    afa_method = RandomWithoutClassifierAFAMethod(
        device=torch.device("cpu"),
        n_classes=train_dataset.label_shape.numel(),
        prob_select_0=0.0
        if contract.soft_budget_param is None
        else contract.soft_budget_param,
    )
    ...  # evaluate it end to end as a smoke check; dummy methods don't train
    return afa_method
```

`inputs: FitInputs`, from `afabench.fit.inputs.load_inputs`, lazily
loads whatever the contract points to: `inputs.train_dataset()`,
`inputs.val_dataset()`, `inputs.initializer()`, `inputs.unmasker()`,
`inputs.classifier(expected_type)` and, for `TrainingContract`,
`inputs.pretrained_model(expected_type)`.

The script itself owns the lifecycle — seeding, wandb, saving — using the
other two helpers:

```python
# scripts/train_method/random_dummy.py
@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/random_dummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    cfg = cast("RandomDummyTrainConfig", OmegaConf.to_object(cfg))
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["random_dummy"], config=cfg):
        afa_method = train_random_dummy(cfg, inputs)
        save_result(afa_method, cfg, cfg)
```

`fit_run` (from `afabench.fit.run`) checks that the train and val dataset
bundles share a dataset identity, seeds with `contract.seed`,
opens a `WandbMetricLogger` or a `NullMetricLogger` depending on
`contract.use_wandb`, and cleans up CUDA state afterwards; use the yielded
logger's `.log(...)` if your training loop reports metrics. `save_result`
writes `obj` as a bundle to `contract.save_path`, with metadata recording the
stage, the contract values and the config fields the contract doesn't cover,
and with the bundle's provenance record (seed, inputs, `method_name` and
dataset identity; see
[`docs/reference/bundle_format.md`](../reference/bundle_format.md)).

Nothing here is Hydra-specific except `@hydra.main` and `OmegaConf.to_object`;
section 5 covers skipping both.

The YAML config wires in the contract's Hydra groups. See
`extra/conf/components/README.md` for how `initializer`, `unmasker` and
`dataset_key` become plain top-level groups instead of Hydra's usual nested
paths:

```yaml
# extra/conf/scripts/train_method/random_dummy/config.yaml
hydra:
  searchpath:
    - file://extra/conf
    - file://extra/conf/global

defaults:
  - train_random_dummy
  - hydra: custom
  - initializer: ???
  - unmasker: ???
  - dataset_key: ???
  - _self_
  - optional experiment@_global_: ${dataset_key}
  - override hydra/job_logging: output_dir_colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm
```

`train_random_dummy` is the name `store_contract_config` registered; it fills
in all the contract fields, leaving the config otherwise empty since
`RandomDummyTrainConfig` adds none.

If your method class is new, register it in `REGISTERED_CLASSES` in
`afabench/core/registry.py` ([`docs/reference/bundle_format.md`](../reference/bundle_format.md))
so `load_bundle` can reconstruct it; `RandomWithoutClassifierAFAMethod` is
already there as `"RandomWithoutClassifierAFAMethod"`.

## 3. Adding a pretraining stage

A method with a pretraining stage additionally gets a script under
`scripts/pretrain_model/`, configured the same way but with a config
dataclass inheriting `PretrainingContract` instead of `TrainingContract`
(a separate contract, currently without the pretrained-model path and
the budgets — see `afabench/fit/contract.py`). Structure it like the training script in
section 2: a plain `pretrain_*` function taking the config and a
`FitInputs`, called from a script that wraps it in `fit_run` and
`save_result`.

A pretrained model is a separate pipeline-level concept from a method; one
pretrained model can be shared by several method names. For example,
`eddi_builtin` and `eddi_external` both depend on the `pvae` pretrained
model:

```yaml
# extra/workflow/conf/pretrain_mappings/all.yaml
pretrain_mapping:
  pvae:
    pretrain_script_name: "odin"
    pretrain_params: []
```

```yaml
# extra/workflow/conf/method_options/all.yaml
method_options:
  eddi_builtin:
    pretrained_model_name: "pvae"
    train_script_name: "eddi_builtin"
    ...
  eddi_external:
    pretrained_model_name: "pvae"
    train_script_name: "eddi_external"
    ...
```

Add an entry to `pretrain_mapping` naming your pretraining script, then point
every method name that depends on it at that name via
`pretrained_model_name` in `method_options` (section 6). The pipeline runs
the pretraining stage once per `(pretrained_model_name, dataset, dataset
realization, pretrain seed)` and passes the resulting bundle's path as
`pretrained_model_bundle_path` to every training run that needs it.

## 4. Configuring hyperparameters

Hyperparameters beyond the contract are entirely the method author's choice
— the contract says nothing about them. The repo's own methods follow one
convention, recommended as the default: a Hydra experiment file per dataset
key, selected automatically through `optional experiment@_global_:
${dataset_key}` in the root config (already present in the YAML in section
2), so `extra/conf/scripts/train_method/<method>/experiment/<dataset
key>.yaml` only needs to exist for the dataset keys that need
non-default values.

Two rules keep this convention from drifting back into the duplication
`docs/explanation/training_contract_inventory.md` describes:

- **Experiment files may not set contract fields.** Snakemake always passes
  `hard_budget`, `seed`, `device` and the rest on the command line, so a
  value set in an experiment file is silently dead in the pipeline and only
  misleads someone running the script by hand. `test/workflow/test_experiment_files_omit_contract_fields.py`
  enforces this for every method; add your method's name there if you
  intentionally leave this for later (see `NOT_YET_PORTED` in that file for
  the pattern), but satisfy it before considering the method done.
- **Deduplicate across dataset keys through shared family files**, pulled in
  via the experiment file's own `defaults:` list, rather than copying the
  same hyperparameters into every dataset key's file. Group by whatever the
  hyperparameters actually vary with (for example tabular vs. image
  datasets), not by listing every dataset key's file by hand.

You are free to configure hyperparameters a different way (plain Python
constants, a JSON file you load yourself, environment variables); the only
constraint is the one in section 5.

## 5. Not using the helpers or Hydra

None of `TrainingContract`, `load_inputs`, `fit_run`, `save_result` or
Hydra is required. If you'd rather write a training script from scratch, it
still has to:

- Parse the training (or pretraining) contract's arguments off
  `sys.argv`, in the plain `key=value` form Snakemake passes them
  (`afabench/fit/contract.py` lists the fields; nothing requires you to
  use the dataclass to hold them).
- Call `afabench.core.bundle_system.bundle.save_bundle` to write a loadable
  bundle to the `save_path` argument, with your method's class registered in
  `REGISTERED_CLASSES` (`afabench/core/registry.py`,
  [`docs/reference/bundle_format.md`](../reference/bundle_format.md)).
  `save_bundle` requires a provenance record; build it with
  `afabench.core.provenance.capture_provenance` from the contract's seed,
  `method_name` and input bundles (`bundle_input`), copying the dataset
  identity from the dataset bundles' records (`bundle_provenance`,
  `shared_dataset_identity`), as `save_result` does.

Everything else — seeding, logging, hyperparameter configuration, smoke-test
handling — is on you, exactly as it would be for any other script.

## 6. Registering with the pipeline

The pipeline doesn't yet know your method exists even once its scripts work
standalone. Four config groups under `extra/workflow/conf/` wire it in,
keyed by a pipeline-level method name (distinct from your training script's
file name):

- **`methods/<variant>.yaml`**: list your method name so the pipeline trains
  and evaluates it.

  ```yaml
  methods:
    - example_method
  ```

- **`method_options/<variant>.yaml`**: everything the pipeline needs to run
  your method — which training script to call, which pretrained model (if
  any) it depends on, evaluation batch size, and which datasets to skip for
  hard- or soft-budget evaluation.

  ```yaml
  method_options:
    example_method:
      pretrained_model_name: "example_model"  # omit if there's no pretraining stage
      train_script_name: "example"
      eval_batch_size:
        default: 128
      hard_budget_ignored_datasets: [imagenette]
      soft_budget_ignored_datasets: [imagenette]
  ```

- **`soft_budget_params/<variant>.yaml`**: the soft-budget parameter values
  to train and evaluate at, as `[train_soft_budget_param,
  eval_soft_budget_param]` pairs; `null` in either position means that stage
  doesn't receive it.

  ```yaml
  soft_budget_params:
    example_method:
      default:
        - [0.1, null]
        - [0.2, null]
  ```

- **`method_sets/<variant>.yaml`**: add your method name to whichever named
  plot groups it belongs in (for example `main`).

If your method introduces a new `AFAMethod` or `AFAClassifier` class, add it
to `REGISTERED_CLASSES` in `afabench/core/registry.py` — this is the same
registration the bundle system needs (section 2 and
[`docs/reference/bundle_format.md`](../reference/bundle_format.md)); there's no separate pipeline
registry.

To also show up in plots, add it to `method_name_mapping` in
`extra/conf/scripts/plotting/common/default.yaml`.

## 7. Testing

`test/scripts/test_training_contract_conformance.py` is the contract
conformance test: for every method name in `method_options`, it runs your
training (and pretraining, if any) script as a subprocess on a generated
smoke CUBE dataset with contract arguments, and asserts a loadable bundle
ends up at `save_path`. It's marked `pipeline` (skipped by default) except
for one always-run `random_dummy` case, so `just qa` exercises the contract
without running every method on every commit.

Once your method is registered in `method_options` (section 6), run its
case directly:

```shell
uv run pytest test/scripts/test_training_contract_conformance.py -m pipeline -k example_method
```

This is the check that actually proves your script satisfies the hard rule
in section 1, independent of whether you used the helpers, Hydra, or neither.

## Running the pipeline

Run the pipeline locally with only your new method and a couple of datasets:

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/all \
    all \
    --config \
      "methods=[example_method]" \
      "datasets=[cube, actg]" \
    --jobs 8
```
