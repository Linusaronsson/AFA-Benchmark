# Training-contract duplication inventory

Investigation for issue #43. Snapshot of `5888192d`. The resulting decision is
`docs/adr/0001-training-contract-as-library.md`. Counts were produced with the
commands in the appendix and can be re-run.

## The contract

The pipeline passes the same arguments to every pretraining and training
script. They are the **training contract** (pretraining uses a subset).

| Field                          | Training | Pretraining | Source in Snakemake      |
| ------------------------------ | -------- | ----------- | ------------------------ |
| `train_dataset_bundle_path`    | yes      | yes         | dataset rule output      |
| `val_dataset_bundle_path`      | yes      | yes         | dataset rule output      |
| `classifier_bundle_path`       | yes      | yes         | external or method classifier |
| `pretrained_model_bundle_path` | if pretrained | no     | `pretrain_model` output  |
| `save_path`                    | yes      | yes         | rule output              |
| `initializer`                  | yes      | yes         | `INITIALIZER`            |
| `unmasker`                     | yes      | yes         | `UNMASKERS[dataset]`     |
| `hard_budget`                  | yes      | no          | `train_hard_budget`      |
| `soft_budget_param`            | yes      | no          | `train_soft_budget_param`|
| `device`                       | yes      | yes         | `DEVICE`                 |
| `seed`                         | yes      | yes         | `train_seed` / `pretrain_seed` |
| `use_wandb`                    | yes      | yes         | `USE_WANDB`              |
| `smoke_test`                   | yes      | yes         | `SMOKE_TEST`             |
| `experiment@_global_`          | yes      | yes         | `{dataset}` (selects per-dataset YAML) |

13 fields for training, 10 for pretraining, plus the dataset key that picks
the per-dataset hyperparameter file.

## Where it is re-declared

| Layer | Files | Contract re-declarations |
| ----- | ----- | ------------------------ |
| Entry-point scripts | 13 `scripts/train_method/*.py`, 6 `scripts/pretrain_model/*.py` (2,362 lines) | each reads the fields off its own config type |
| Config dataclasses | 17 classes in `afabench/components/methods/**/config.py` (12 training, 5 pretraining; AACO pretraining reuses `AACOTrainConfig`) | 201 field declarations |
| Hydra root configs | 19 `extra/conf/scripts/{train_method,pretrain_model}/*/config.yaml` (808 lines) | 11 contract keys in each of 19 files, `pretrained_model_bundle_path` in 8 |
| Hydra experiment files | 115 training + 64 pretraining `experiment/<dataset>.yaml` | 58 hardcode `hard_budget` (see below) |
| Snakemake rules | `extra/workflow/snakefiles/rules/training.smk` | 3 rules with the same shell body: `pretrain_model`, `train_method_with_pretrained_model`, `train_method_without_pretrained_model` |
| Workflow config | `extra/workflow/src/config.py`, `conf/method_options/*.yaml`, `conf/pretrain_mappings/*.yaml` | method → script name, pretrained model name, free-form `method_specific_params` / `pretrain_params` strings |
| Tutorial | `docs/how-to/add_method.md` (390 lines) | restates the contract fields and both YAML skeletons |

Per-field count in the 17 config dataclasses:

| Field | Classes |
| ----- | ------- |
| `train_dataset_bundle_path`, `val_dataset_bundle_path`, `classifier_bundle_path`, `save_path`, `initializer`, `unmasker`, `device`, `seed`, `use_wandb`, `smoke_test` | 17 each |
| `hard_budget`, `soft_budget_param` | 12 each |
| `pretrained_model_bundle_path` | 7 |

The types also drift. `seed` is `int | None` in most classes and `int = 42` in
AACO. `device` is `str`, `str | None` or `str = "cpu"`. Paths are `str` in most
classes and `Path | None` in AACO. `classifier_bundle_path` is required in
some classes and `None` in others, even though the pipeline always passes it.
Eight files carry the comment "not needed for this method, but pipeline passes
it to us".

## Lifecycle work repeated per method

What each entry point does besides the method's own training. "lib" means the
work happens inside a library `train_*` function rather than the script.

| Entry point | Seeding | Smoke-test overrides | wandb init | Saves bundle |
| ----------- | ------- | -------------------- | ---------- | ------------ |
| `train_method/aaco` | lib | lib | script | lib |
| `train_method/aaco_nn` | script | script | script | script |
| `train_method/cae` | lib | lib | script | lib |
| `train_method/dime`, `gdfs` | lib | lib | script | lib |
| `train_method/eddi_builtin`, `eddi_external` | script | none | script | script |
| `train_method/jafa`, `odin`, `ol` | script | script | `RLTrainer` | `RLTrainer` |
| `train_method/permutation` | script | script | script | script |
| `train_method/random_dummy`, `sequential_dummy` | script | script | script | script |
| `pretrain_model/aaco` | lib | lib | script | lib |
| `pretrain_model/dime`, `gdfs` | lib | lib | script | lib |
| `pretrain_model/jafa`, `odin`, `ol` | script | script | script | script |

So seeding, smoke-test handling and saving each live in about four places, and
the place depends on which method family you are reading.

## Near-duplicate entry points

Line diffs between scripts (lines that differ, `diff | grep '^[<>]'`):

| Pair | Differing lines | What differs |
| ---- | --------------- | ------------ |
| `train_method/eddi_builtin` vs `eddi_external` | 6 | config path, wandb tag, `classifier_bundle_path=None` vs the path |
| `pretrain_model/aaco` vs `train_method/aaco` | same body | config path, wandb job type, one log line; both call `aaco.train.run(cfg)` |
| `train_method/dime` vs `gdfs` | 20 | renames only |
| `train_method/cae` vs `dime` / `gdfs` | 22 | renames only |
| `pretrain_model/dime` vs `gdfs` | renames only | same shape as training |
| `train_method/random_dummy` vs `sequential_dummy` | 29 | method class, wandb tag, one constructor argument |

## The hard-budget alias workaround

Snakemake passes a top-level `hard_budget=`. The RL methods read the budget
from `mdp.hard_budget` instead, so each RL training config declares both and
the script copies one onto the other:

- `scripts/train_method/jafa.py:42`, `odin.py:43`, `ol.py:97`:
  `cfg.mdp.hard_budget = cfg.hard_budget`
- `extra/conf/scripts/train_method/{jafa,odin,ol}/config.yaml`:
  `mdp.hard_budget: ???` plus a top-level `hard_budget: null` under
  "Alias arguments, only to implement interface assumed by snakemake".

The issue names JAFA, but all three RL training entry points do this.

## Per-dataset hyperparameter layout

Every root config ends with `optional experiment@_global_: ???`, and every
Snakemake rule passes `experiment@_global_={dataset}`. Each method therefore
has an `experiment/` directory with one file per dataset it overrides.

| Entry point | Experiment files | Distinct contents |
| ----------- | ---------------- | ----------------- |
| `train_method/aaco` | 10 | 10 |
| `train_method/aaco_nn` | 16 | 1 |
| `train_method/cae` | 13 | 2 |
| `train_method/dime` | 16 | 7 |
| `train_method/eddi_builtin` | 13 | 1 |
| `train_method/eddi_external` | 13 | 1 |
| `train_method/gdfs` | 16 | 7 |
| `train_method/ol` | 5 | 1 |
| `train_method/permutation` | 13 | 2 |
| `train_method/jafa`, `odin`, `random_dummy`, `sequential_dummy` | 0 | 0 |
| `pretrain_model/aaco` | 15 | 10 |
| `pretrain_model/dime` | 16 | 6 |
| `pretrain_model/gdfs` | 16 | 6 |
| `pretrain_model/jafa` | 5 | 2 |
| `pretrain_model/odin` | 7 | 2 |
| `pretrain_model/ol` | 5 | 1 |
| **Total** | **179** | **59** |

About two thirds of the files (120 of 179) repeat content that another file in the same directory
already has. The variation is usually by dataset family (tabular vs image, or
the CUBE-NM variants), not by individual dataset.

Experiment files also override contract fields, which hides bugs:

- 58 files under `train_method/{cae,dime,gdfs,permutation}/experiment/` set
  `hard_budget`. Snakemake always overrides it on the command line, so the
  value is dead in the pipeline but live when a developer runs the script by
  hand.
- `train_method/aaco/experiment/*.yaml` set `seed: 42` and `device: "cpu"`,
  and some still reference legacy wandb `dataset_artifact_name` values.

## Smaller inconsistencies found on the way

- `random_dummy` and `sequential_dummy` log their training run with wandb
  `job_type="pretraining"`.
- Bundle metadata takes three shapes: `{"config": asdict(cfg)}`, a hand-built
  dict (dummy methods, AACO, AACO+NN) or a pretraining-specific dict. No code
  downstream reads it.
- `afabench/core/registry.py` is a class-name → import-path table used by
  bundle loading. It has no decorator-based `Registry[T]`, although
  `AGENTS.md` described one (since corrected). Any "resolve the trainer through the registry"
  design needs to add that mechanism or extend the table.
- The only training-adjacent tests are `test/scripts/test_train_method_smoke_test.py`
  (helpers in `afabench.training.smoke_test`) and
  `test/src/afa_rl/test_training.py`. No test trains a method end to end.
  `pytest.ini` declares a `pipeline` marker that no test uses.

## Cost of adding a method today

Adding a method with a pretraining stage touches:

1. `scripts/pretrain_model/<m>.py`
2. `scripts/train_method/<m>.py`
3. `afabench/components/methods/<family>/<m>/config.py` (two dataclasses that
   re-declare the contract)
4. `extra/conf/scripts/pretrain_model/<m>/config.yaml`
5. `extra/conf/scripts/train_method/<m>/config.yaml`
6. `extra/conf/scripts/pretrain_model/<m>/experiment/<dataset>.yaml` per
   dataset that differs
7. `extra/conf/scripts/train_method/<m>/experiment/<dataset>.yaml` per
   dataset that differs
8. `extra/workflow/conf/pretrain_mappings/{all,kdd26}.yaml`
9. `extra/workflow/conf/method_options/{all,kdd26}.yaml`
10. `extra/workflow/conf/methods/*.yaml`
11. `extra/workflow/conf/soft_budget_params/*.yaml`
12. `extra/workflow/conf/method_sets/*.yaml`
13. `afabench/core/registry.py` (method class and any classifier class)
14. The method package itself

## Appendix: commands

```bash
# Config dataclasses that re-declare the contract
uv run python - <<'EOF'
import ast, pathlib, collections
F = ["train_dataset_bundle_path", "val_dataset_bundle_path",
     "classifier_bundle_path", "pretrained_model_bundle_path", "save_path",
     "initializer", "unmasker", "hard_budget", "soft_budget_param",
     "device", "seed", "use_wandb", "smoke_test"]
cnt, n = collections.Counter(), 0
for p in pathlib.Path("afabench/components/methods").rglob("config.py"):
    for c in ast.walk(ast.parse(p.read_text())):
        if isinstance(c, ast.ClassDef):
            names = [x.target.id for x in c.body
                     if isinstance(x, ast.AnnAssign)
                     and isinstance(x.target, ast.Name)]
            if "save_path" in names:
                n += 1
                cnt.update(f for f in F if f in names)
print(n, sum(cnt.values()), dict(cnt))
EOF

# Experiment files and distinct contents per entry point
cd extra/conf/scripts
for m in {train_method,pretrain_model}/*/; do
  files=$(ls "$m"experiment/*.yaml 2>/dev/null)
  echo "$m $(echo "$files" | grep -c .)" \
    "$(for f in $files; do md5sum < "$f"; done | sort -u | wc -l)"
done
```
