# Install and use the library

For the [library user](../../explanation/user_types.md#library-user) who
wants to call AFABench from their own Python code without running the
benchmark pipeline.

## Install into your Python environment

Use Linux and Python **3.12.10**, with Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/) available.
AFABench is not published on PyPI; install from a local clone instead of
running `pip install afabench` against the package index.

Clone the source, then create an environment in your own project directory
(or use your existing Python 3.12.10 environment):

```bash
git clone https://github.com/MachineLearning-DecisionMaking/AFABench.git
cd /path/to/your/project
uv venv --python 3.12.10 .venv
uv pip install --python .venv/bin/python /absolute/path/to/AFABench
```

Replace both paths with your actual directories. Allow space for the ML
and GPU dependencies: library installation currently also installs
Snakemake and other pipeline dependencies. For the installation boundary,
see [Library and benchmark pipeline](../../explanation/user_types.md#library-and-benchmark-pipeline).

Verify the installed distribution and import:

```bash
.venv/bin/python -c 'import afabench; from importlib.metadata import version; print(version("afabench"))'
```

## Evaluate a batch from your own code

Save the following as `evaluate_batch.py` in your project directory. Replace
the demonstration policy, classifier, features and one-hot labels with your
own components when adapting it. The classifier here only applies softmax
to the two masked features; do not use it as a trained benchmark classifier.
Use [class probabilities](../../reference/evaluation_dataframes.md#class-probabilities),
not logits, as the classifier's return value.

```python
import torch

from afabench.components.unmaskers.direct_unmasker import DirectUnmasker
from afabench.core.types import (
    AFAAction,
    ClassProbabilities,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.evaluation.eval import process_batch


def policy(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask,
    label: Label | None = None,
    feature_shape: torch.Size | None = None,
) -> AFAAction:
    available = ~selection_mask
    next_action = available.long().argmax(dim=-1, keepdim=True) + 1
    return torch.where(available.any(dim=-1, keepdim=True), next_action, 0)


def classify(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    label: Label | None = None,
    feature_shape: torch.Size | None = None,
) -> ClassProbabilities:
    return masked_features.softmax(dim=-1)


features = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
initial_mask = torch.zeros_like(features, dtype=torch.bool)
results = process_batch(
    afa_action_fn=policy,
    afa_unmask_fn=DirectUnmasker().unmask,
    n_selection_choices=2,
    features=features,
    initial_feature_mask=initial_mask,
    initial_masked_features=features.masked_fill(~initial_mask, 0.0),
    true_label=labels,
    feature_shape=features.shape[1:],
    external_afa_predict_fn=classify,
    selection_budget=1.0,
    force_acquisition=True,
)
stops = results.loc[results["action_performed"] == 0]
assert len(results) == 4
assert len(stops) == 2
assert stops["accumulated_cost"].eq(1.0).all()
print(results[["episode_id", "step", "action_performed", "accumulated_cost"]])
```

Run it from your project directory:

```bash
.venv/bin/python evaluate_batch.py
```

Check that each episode has action `1` at step `0`, followed by stop action
`0` at step `1`, with accumulated cost `1.0`. Look up the other returned
columns in [Evaluation dataframes](../../reference/evaluation_dataframes.md#contract).
Keep each batch's episode IDs separate when combining results; they are
local to that `process_batch` call.
