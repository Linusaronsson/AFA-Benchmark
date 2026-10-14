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
`src/afabench/fit/contract.py`. Both extend `BaseContract` and are
independent of each other: `TrainingContract` has the pretrained-model path
and the budgets, which `PretrainingContract` doesn't, but the two stages are
free to diverge further. A method whose stages share logic still gives
each stage its own config class (AACO's `AACOPretrainConfig` and
`AACOTrainConfig`) and passes stage-specific values such as budgets to the
shared code as arguments, because the pretraining stage passes no
`method_name` or budgets. Read that file before writing a new method's config.

## 2. Minimal method

The smallest way to satisfy the contract is a training script with no
pretraining stage that inherits `TrainingContract` for its config and uses
the optional library helpers. The ported dummy methods
(`src/afabench/components/methods/dummy/`) are the worked example; this section
walks through `random_dummy`.

The config dataclass adds no fields of its own:

```python
# src/afabench/components/methods/dummy/config.py
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
# src/afabench/components/methods/dummy/train.py (abridged)
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
`inputs.pretrained_model(expected_type)`. The Initializer and Unmasker
come seeded with `contract.seed`.

The script itself owns the lifecycle — seeding, wandb, saving — using the
other two helpers:

```python
# scripts/train_method/random_dummy.py
@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/random_dummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    cfg = cast("RandomDummyTrainConfig", OmegaConf.to_object(cfg))
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["random_dummy"], inputs=inputs):
        afa_method = train_random_dummy(cfg, inputs)
        save_result(afa_method, cfg, inputs=inputs)
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
Both helpers take the method config from the contract, since your config
dataclass inherits it; pass `config=` only if the method config is a
separate object. Pass both helpers the script's `inputs`, so that each
input bundle's manifest is read once; without `inputs=` they read the
manifests again.

To make a smoke test (`contract.smoke_test`) fast, shrink the config with
`limit` from `afabench.fit.smoke_test` before loading the inputs:
`cfg = limit(cfg, nepochs=1, patience=1)` replaces those fields during a
smoke test and returns `cfg` unchanged otherwise. A mapping replaces only
the named fields of a nested config dataclass, as in
`limit(cfg, selector={"nepochs": 1})`.

Nothing here is Hydra-specific except `@hydra.main` and `OmegaConf.to_object`;
section 6 covers skipping both.

The YAML config wires in the contract's Hydra groups. See
`conf/components/README.md` for how `initializer`, `unmasker` and
`dataset_key` become plain top-level groups instead of Hydra's usual nested
paths:

```yaml
# conf/scripts/train_method/random_dummy/config.yaml
hydra:
  searchpath:
    - file://conf
    - file://conf/global

defaults:
  - train_random_dummy
  - hydra: custom
  - initializer: ???
  - unmasker: ???
  - dataset_key: ???
  - _self_
  - optional dataset_override@_global_: ${dataset_key}
  - override hydra/job_logging: output_dir_colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm
```

`train_random_dummy` is the name `store_contract_config` registered; it fills
in all the contract fields, leaving the config otherwise empty since
`RandomDummyTrainConfig` adds none.

If your method class is new, register it in `REGISTERED_CLASSES` in
`src/afabench/core/registry.py` ([`docs/reference/bundle_format.md`](../reference/bundle_format.md))
so `load_bundle` can reconstruct it; `RandomWithoutClassifierAFAMethod` is
already there as `"RandomWithoutClassifierAFAMethod"`.

## 3. The method class

The object the training script returns and saves is the method class. It
implements the `AFAMethod` protocol in `src/afabench/core/types.py`, which
is what evaluation and the bundle system call. Build it on `AFAMethodBase`
from `afabench.components.methods.common`, as every method in the repo
does, so you write only what is specific to your method:

- Set the device with `super().__init__(device)`. The base's `to` then
  moves every tensor, `nn.Module` and `AFAClassifier` attribute, and
  `device` reports where the method computes. If you keep modules in a
  container such as a dict, extend `to` and call `super().to(device)`
  first.
- Declare `has_builtin_classifier`. It is abstract, so a method that
  forgets it does not instantiate. True means evaluation also reports
  `predict` as the method's built-in classifier.
- Implement `act` and `predict`, decorated with `on_method_device`: the
  body then runs on the method's device whatever device the caller's
  tensors are on, and the answer comes back on the caller's.
- Implement `save` and `load` in whatever layout suits the method; `load`
  takes the device to place the method on. A method that ships a
  classifier can store it with `save_classifier` and `load_classifier`,
  which write `classifier.pt` and `classifier_class_name.txt` under the
  bundle's data folder.
- Override `cost_param` and `set_cost_param` if the method has a
  soft-budget parameter, and `set_seed` if it has a generator of its own.
  The base's defaults are None and no-ops; evaluation seeds torch's global
  generators before it calls `set_seed`.

`RandomWithClassifierAFAMethod` in
`src/afabench/components/methods/dummy/with_classifier.py` is the shortest
complete example. Register the class in `REGISTERED_CLASSES` as section 2
says.

## 4. Adding a pretraining stage

A method with a pretraining stage additionally gets a script under
`scripts/pretrain_model/`, configured the same way but with a config
dataclass inheriting `PretrainingContract` instead of `TrainingContract`
(a separate contract, currently without the pretrained-model path and
the budgets — see `src/afabench/fit/contract.py`). Structure it like the training script in
section 2: a plain `pretrain_*` function taking the config and a
`FitInputs`, called from a script that wraps it in `fit_run` and
`save_result`.

If the pretrained model is a Lightning module trained by supervised
learning on the train and val datasets, inherit
`SupervisedPretrainingConfig` from `afabench.fit.config` instead, and let
the script's `main` call `supervised_pretraining_main` from
`afabench.fit.supervised_learning` with a function that builds the module
from the config and the `FitInputs`. It applies the smoke-test limits,
trains and saves; `scripts/pretrain_model/jafa.py` is an example.

A pretrained model is a separate pipeline-level concept from a method; one
pretrained model can be shared by several method names. For example,
`eddi_builtin` and `eddi_external` both depend on the `pvae` pretrained
model:

```yaml
# workflow/conf/pretrain_mappings/all.yaml
pretrain_mapping:
  pvae:
    pretrain_script_name: "odin"
    pretrain_params: []
```

```yaml
# workflow/conf/method_options/all.yaml
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
`pretrained_model_name` in `method_options` (section 7). The pipeline runs
the pretraining stage once per `(pretrained_model_name, dataset, dataset
realization, pretrain seed)` and passes the resulting bundle's path as
`pretrained_model_bundle_path` to every training run that needs it.

## 5. Configuring hyperparameters

Hyperparameters beyond the contract are entirely the method author's choice
— the contract says nothing about them. The repo's own methods follow one
convention, recommended as the default: the root config gives every
hyperparameter a default, and a dataset override,
`conf/scripts/train_method/<method>/dataset_override/<dataset key>.yaml`, changes
some of them on one dataset key. It is selected automatically through
`optional dataset_override@_global_: ${dataset_key}` in the root config (already
present in the YAML in section 2), so it only exists for dataset keys that
need other values. A dataset key without one runs on the defaults, so a new
dataset never fails for lack of a file;
`test/workflow/test_pipeline_script_configs_complete.py` fails on any
hyperparameter left without a default.

A method with tabular and image networks keeps both architectures in its
config and an `architecture_kind` choosing between them (DIME, GDFS and
CAE), rather than a Hydra group per architecture: an override cannot
select a group the root config already selects (see the gotchas in
`conf/components/README.md`).

Three rules keep this convention from drifting back into the duplication
`docs/explanation/training_contract_inventory.md` describes:

- **Overrides hold only what differs from the defaults.** A file that
  restates the defaults hides which datasets actually deviate.

- **Dataset overrides may not set contract fields.** Snakemake always passes
  `hard_budget`, `seed`, `device` and the rest on the command line, so a
  value set in a dataset override is silently dead in the pipeline and only
  misleads someone running the script by hand. `test/workflow/test_dataset_overrides_omit_contract_fields.py`
  enforces this for every method; add your method's name there if you
  intentionally leave this for later (see `NOT_YET_PORTED` in that file for
  the pattern), but satisfy it before considering the method done.
- **Deduplicate across dataset keys through shared family files**, pulled in
  via the dataset override's own `defaults:` list, rather than copying the
  same hyperparameters into every dataset key's file. Group by whatever the
  hyperparameters actually vary with (for example tabular vs. image
  datasets), not by listing every dataset key's file by hand.

You are free to configure hyperparameters a different way (plain Python
constants, a JSON file you load yourself, environment variables); the only
constraint is the one in section 6.

## 6. Not using the helpers or Hydra

None of `TrainingContract`, `load_inputs`, `fit_run`, `save_result` or
Hydra is required. If you'd rather write a training script from scratch, it
still has to:

- Parse the training (or pretraining) contract's arguments off
  `sys.argv`, in the plain `key=value` form Snakemake passes them
  (`src/afabench/fit/contract.py` lists the fields; nothing requires you to
  use the dataclass to hold them).
- Call `afabench.core.bundle_system.bundle.save_bundle` to write a loadable
  bundle to the `save_path` argument, with your method's class registered in
  `REGISTERED_CLASSES` (`src/afabench/core/registry.py`,
  [`docs/reference/bundle_format.md`](../reference/bundle_format.md)).
  `save_bundle` requires a provenance record; build it with
  `afabench.core.provenance.capture_provenance` from the contract's seed,
  `method_name` and input bundles (`bundle_input`), copying the dataset
  identity from the dataset bundles' records
  (`shared_bundle_dataset_identity`), as `save_result` does.

Everything else — seeding, logging, hyperparameter configuration, smoke-test
handling — is on you, exactly as it would be for any other script.

## 7. Registering with the pipeline

The pipeline doesn't yet know your method exists even once its scripts work
standalone. Four config groups under `workflow/conf/` wire it in,
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
  any) it depends on, evaluation batch size, which datasets to skip for
  hard-budget evaluation, and which datasets to run soft-budget evaluation
  on: soft budgets are opt-in, so a dataset not listed gets none.

  ```yaml
  method_options:
    example_method:
      pretrained_model_name: "example_model"  # omit if there's no pretraining stage
      train_script_name: "example"
      eval_batch_size:
        default: 128
      hard_budget_ignored_datasets: [imagenette]
      soft_budget_datasets: [cube, miniboone]
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

- **`method_sets/<variant>.yaml`** (optional): add your method name to
  whichever named plot groups it belongs in (for example `main`). It is
  plotted beside every other method of the run in the `all` method set
  either way.

If your method introduces a new `AFAMethod` or `AFAClassifier` class, add it
to `REGISTERED_CLASSES` in `src/afabench/core/registry.py` — this is the same
registration the bundle system needs (section 2 and
[`docs/reference/bundle_format.md`](../reference/bundle_format.md)); there's no separate pipeline
registry.

To also show up in plots, add it to `method_name_mapping` in
`conf/scripts/plotting/common/default.yaml`.

## 8. Testing

`test/scripts/test_training_contract_conformance.py` is the contract
conformance test: for every method name in `method_options`, it runs your
training (and pretraining, if any) script as a subprocess on a generated
smoke CUBE dataset with contract arguments, and asserts a loadable bundle
ends up at `save_path`. It's marked `pipeline` (skipped by default) except
for one always-run `random_dummy` case, so `just qa` exercises the contract
without running every method on every commit.

Once your method is registered in `method_options` (section 7), run its
case directly:

```shell
uv run pytest test/scripts/test_training_contract_conformance.py -m pipeline -k example_method
```

This is the check that actually proves your script satisfies the hard rule
in section 1, independent of whether you used the helpers, Hydra, or neither.

## Running the pipeline

Run the pipeline locally with only your new method and a couple of datasets:

```shell
workflow/run.py --jobs 8 \
    "methods=[example_method]" \
    "datasets=[cube, actg]" \
    execution_file=workflow/profiles/execution/cpu.yaml
```
