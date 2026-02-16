# Adding a New Method

## Overview

To add a new Active Feature Acquisition (AFA) method to the benchmark, you need to:
1. Create a training script that implements your method
2. (Optional) Create a pretraining script if your method requires pre-trained models
3. Add configuration classes and YAML files for Hydra
4. Register your method with the pipeline

This tutorial assumes your method requires a pretraining stage. Our example method will be called `Example` and will output random actions. You can reference the `RandomDummyAFAMethod` implementation in the codebase to understand the AFA method interface.

## Step-by-Step Guide

### 1. Pretraining a model

Create a pretraining script at `scripts/pretrain/example.py`:
```python
import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import OmegaConf
from torch import nn

from afabench.common.bundle import save_bundle
from afabench.common.config_classes import (
    ExamplePretrainConfig,
)
from afabench.common.torch_bundle import TorchModelBundle
from afabench.common.utils import (
    initialize_wandb_run,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/example",
    config_name="config",
)
def main(cfg: ExamplePretrainConfig) -> None:
    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=cast(
                "dict[str,Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="pretraining",
            tags=["example"],
        )
    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")

    # This is where training should be done, but we just save a random linear layer
    save_bundle(
        TorchModelBundle(nn.Linear(1, 1)), Path(cfg.save_path), metadata={}
    )


if __name__ == "__main__":
    main()
```

See the `pretrain_model` rule in `training.smk` for arguments that the script is required to support. Since the arguments are passed without dashes, you are encouraged to use Hydra for the script configuration. Place your pretrain configuration class in `afabench/common/config._classes.py`:

```python
@dataclass
class ExamplePretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: (
        str | None
    )  # not needed for this method, but pipeline passes it to us
    save_path: str
    device: str
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False

cs.store(name="pretrain_example", node=ExamplePretrainConfig)
```

You also need to configure (optional) default values for the script, which you do in `extra/conf/scripts/pretrain/example/config.yaml`:
```yaml
hydra:
  searchpath:
    - file://extra/conf
    - file://extra/conf/global

defaults:
  - hydra: custom
  - _self_
  - optional /components/initializers@initializer: ???
  - /components/unmaskers@unmasker: ???
  - optional experiment@_global_: ???
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm


train_dataset_bundle_path: ???
val_dataset_bundle_path: ???
classifier_bundle_path: null # not needed for this method, but pipeline passes it to us
save_path: ???
device: cuda
seed: null
use_wandb: false
smoke_test: false
```

The pipeline has its own concept of what a "pretrained model" is, so you should update `extra/workflow/conf/pretrain_mapping.yaml`:
```yaml
pretrain_mapping:
    example_model:
        pretrain_script_name: "example"
        pretrain_params: []
    # other models...
```

This defines a pretrained model called `example_model`, which your method will later depend on. The `pretrain_script_name` refers to files in `scripts/pretrain/`.

### 2. Training a method

You cannot yet run the pretraining stage, since the pipeline works backwards from which *methods* need to be trained.

Add a training script at `scripts/train/example.py`:
```python
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.afa_methods import RandomDummyAFAMethod
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    ExampleTrainConfig,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    initialize_wandb_run,
    set_seed,
)
from afabench.eval.eval import eval_afa_method

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/example",
    config_name="config",
)
def main(cfg: ExampleTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cast(
                "dict[str,Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="training",
            tags=["example"],
        )
    else:
        run = None

    if cfg.smoke_test:
        log.info("Smoke test detected.")
        # Because this method does not train, smoke test is no different

    train_dataset, dataset_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))

    assert len(train_dataset.label_shape) == 1, "Only 1D labels supported"

    # Create initializer
    initializer = get_afa_initializer_from_config(cfg.initializer)

    # Create unmasker
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)

    # Usually, this is where training would happen

    afa_method = RandomDummyAFAMethod(
        device=torch.device(cfg.device),
        n_classes=train_dataset.label_shape.numel(),
        prob_select_0=0.0
        if cfg.soft_budget_param is None
        else cfg.soft_budget_param,
    )

    # Save method as a bundle
    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={
            "dataset_class_name": dataset_manifest["class_name"],
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            "seed": cfg.seed,
            "soft_budget_param": cfg.soft_budget_param,
            "hard_budget": cfg.hard_budget,
            "initializer_class_name": cfg.initializer.class_name,
            "unmasker_class_name": cfg.unmasker.class_name,
        },
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
```

Note that this is almost identical to `scripts/train/random_dummy.py`. Had we used a different class than `RandomAFAMethod` to represent our method, we would also have to add it to the registry object `REGISTERED_CLASSES` in `afabench/common/registry.py`. However, there is already an entry for `"RandomDummyAFAMethod": "afabench.common.afa_methods.RandomDummyAFAMethod"`.

Similar to before, you have to configure a config dataclass in `config_classes.py`:
```python
@dataclass
class ExampleTrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: (
        str | None
    )  # not needed for this method, but pipeline passes it to us
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    hard_budget: int | None  # not used, but pretend that it is
    soft_budget_param: float | None

    device: str
    seed: int | None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_example", node=ExampleTrainConfig)
```
and also create an instance of it at `extra/conf/scripts/train/example/config.yaml`:
```yaml
hydra:
  searchpath:
    - file://extra/conf
    - file://extra/conf/global

defaults:
  - hydra: custom
  - /components/initializers@initializer: ???
  - /components/unmaskers@unmasker: ???
  - _self_
  - optional experiment@_global_: ???
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/launcher: custom_slurm

train_dataset_bundle_path: ???
val_dataset_bundle_path: ???
pretrained_model_bundle_path: ???
classifier_bundle_path: null # not needed for this method, but pipeline passes it to us
save_path: ???
# initializer set as component
# unmasker set as component
hard_budget: null
soft_budget_param: null

device: cpu
seed: null
use_wandb: false
smoke_test: false
```

## Integrating method in pipeline

So far, we have only created scripts that work individually, but these scripts will not yet run automatically when we execute the pipeline. `extra/workflow/conf/methods.yaml` contains a list of all methods that the pipeline will run. Let's call the new method `example_method`, so add `- example_method` as a new line.

Next, add your method options to `extra/workflow/conf/method_options.yaml`.
```yaml
example_method:
  pretrained_model_name: "example_model"
  train_script_name: "example"
  eval_batch_size:
    default: 128
  hard_budget_ignored_datasets: [imagenette]
  soft_budget_ignored_datasets: [imagenette]
```

This assumes that the method supports training on all datasets except Imagenette (which uses image patches).

### 3. Configuring soft budgets

We need to define reasonable soft-budget parameters for our method, which we do in `extra/workflow/conf/soft_budget_params.yaml`. For this method, the soft-budget parameter corresponds to the probability of choosing the stop action. For simplicity, let us define values that are reused across all datasets:
```yaml
soft_budget_params:
  example_method:
    default:
      - [0.1, null]
      - [0.2, null]
      - [0.3, null]
  # more methods...
```

This declaration ensures that soft-budget parameters are only passed to the method during training, not during evaluation.

### 4. Visualization options

Choose how the method should be displayed in plots by adding an entry to `METHOD_NAME_MAPPING` in `afabench/eval/plotting_config.py`. For example, we may add
```python
"example_method": "Example"
```

Also decide if you want to compare the method with any specific other methods. In this case, we decide to only add it to the main results plot, so we add it to the `main` method set in `extra/workflow/conf/method_sets.yaml`.
```yaml
method_sets:
    main:
        # other methods...
        - example_method
         # more methods...
```

### 5. Running the pipeline

Let us run the pipeline locally using 8 cores, with only the new method and two datasets:
```shell
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
      methods=[example_method] \
      datasets=[cube, actg] \
      eval_dataset_split=val \
      dataset_instance_indices=[0,1] \
      smoke_test=true \
      use_wandb=false \
      device=cpu \
    --jobs 8 \
    --keep-going
```
