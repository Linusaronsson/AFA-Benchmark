---
status: accepted
---

# Artifacts record their own provenance

> Note (#71): "dataset instance" was renamed to **dataset realization**;
> `dataset_instance_index` below is now `dataset_realization_index`.

Today an artifact's identity lives in its Snakemake path. An artifact is a
bundle or an evaluation table (see **Artifact** in `CONTEXT.md`). Nothing inside a
bundle or an evaluation Parquet file records the producing code commit, the
resolved configuration, the input bundles, or the seed actually used. The
evaluator writes only `eval_seed` and `eval_hard_budget`; method name, dataset
key, training seed and budgets are re-attached by the transform step from
Snakemake wildcards, and dataset instance index and evaluation split are not
attached at all. An evaluation run by hand yields an anonymous file, and a
benchmark release (#36) cannot describe a bundle once it leaves its path.

We fix this with a typed **provenance record**, captured once per artifact
by the library code that writes it, embedded in every bundle manifest and in
every raw evaluation table, and propagated from inputs to outputs rather than
reconstructed from paths. Issue #45 holds the investigation.

## Considered options

- **Paths plus a release manifest only.** Keep identity in the Snakemake
  layout and let the release manifest (#63) describe it. Rejected: hand runs
  stay anonymous, the path layout becomes an undocumented schema, and a
  published bundle loses its identity once restored elsewhere.
- **Sidecar `provenance.json` beside each artifact.** Rejected: bundles
  already carry a manifest and Parquet already supports key-value metadata;
  a sidecar is separated from its artifact by any copy that forgets it.
- **Conventions inside the free-form `metadata` dictionary.** Rejected: this
  is how the drift happened. The two dataset generators already write
  different keys for the same facts (`kwargs` vs `dataset_kwargs`, `split`
  only for image datasets).
- **A typed record in the manifest and in Parquet, plus identity columns.**
  Chosen.

## The provenance record

A frozen dataclass serialised as a JSON object. `null` always means
"unknown", never a default.

| Field | Type | Meaning |
| --- | --- | --- |
| `provenance_version` | int | Schema version, starting at 1. Readers reject versions they do not know. |
| `stage` | string | The pipeline stage that produced the artifact: `dataset_generation`, `classifier_training`, `pretraining`, `training` or `evaluation`. |
| `created_at` | string | UTC ISO-8601 time of capture. |
| `code_commit` | string or null | `git rev-parse HEAD` of the checkout; null outside a git work tree. |
| `code_dirty` | bool or null | Whether tracked files differ from `code_commit`. Untracked files (outputs, data) are ignored. The diff itself is not recorded. |
| `resolved_config` | object | The script's full configuration after interpolation and after smoke-test overrides, as the script actually used it. Must be JSON-serialisable; otherwise capture raises. |
| `seed` | int | The seed actually used, never null (see the reproducibility policy). |
| `smoke_test` | bool | Promoted from the config because release tooling must refuse smoke artifacts. |
| `method_name` | string or null | Pipeline method name for training and evaluation records; null for other stages. |
| `dataset_key` | string or null | Dataset key. |
| `dataset_instance_index` | int or null | Dataset instance index. |
| `split` | string or null | `train`, `val` or `test` for a dataset bundle (its own split) and for an evaluation (the evaluated split); null otherwise. |
| `inputs` | list | One entry per input bundle: `role` (`train_dataset`, `val_dataset`, `eval_dataset`, `classifier`, `pretrained_model`, `method`), `path` as given, `class_name`, and `content_hash` copied from the input's manifest (null for inputs that predate this ADR). |
| `environment` | object | Python version; versions of `afabench`, `torch`, `numpy` and `pandas`; SHA-256 of `uv.lock` (null if absent); `platform.platform()`. The lockfile hash identifies the full environment without recording a package freeze. |
| `compute` | object | Device as configured; accelerator name (CUDA device name, null on CPU); torch CUDA and cuDNN versions; `torch.get_float32_matmul_precision()`; cuDNN `deterministic` and `benchmark` flags; whether deterministic algorithms are enabled. |

Paths are recorded exactly as passed, which is relative under the pipeline.
Release tooling may redact absolute paths before publishing; the record
itself is not rewritten.

## Where provenance lives

- **Bundles.** The manifest gains two top-level keys beside `metadata`:
  `provenance` (the record) and `content_hash` (`sha256:<hex>` over the
  `data/` folder's files in sorted relative-path order, excluding the
  manifest). `bundle_version` goes from `1.0.0` to `1.1.0`; manifests
  without the new keys still load. `metadata` stays free-form and
  object-specific, and `save_result` keeps the shape fixed by ADR 0001.
- **Raw evaluation tables.** The record is stored as JSON under the Arrow
  schema metadata key `afabench.provenance`, and the identity it implies is
  also written as columns (below). Both are needed: pandas drops custom
  schema metadata on read and concatenation, and results-only users
  read columns with plain Parquet readers.
- **Transformed tables** copy the source record into their own schema
  metadata (one raw table in, one transformed table out). Merged tables and
  plots carry none; their rows keep the identity columns and the release
  manifest (#63) collects the records.

Consumers read an input's `content_hash` from its manifest instead of
rehashing it. Verifying hashes against bundle contents is the job of the
publish and download tooling (#36), not of `load_bundle`. For image datasets
the bundle holds split indices and configuration, not pixels, so the hash
identifies those and the source files are identified only through the
config.

## Where capture happens

One HF-unaware library module builds the record. It collects the
environment, compute and code facts itself; callers supply the stage,
resolved config, resolved seed, inputs and identity. `save_bundle` requires
a record (keyword-only), so a bundle without provenance cannot be written
by production code; tests build records through the same module. Training,
evaluation and plotting scripts never learn about Hugging Face: publishing
only reads these records.

| Stage | Captured by | Seed | Inputs | Identity |
| --- | --- | --- | --- | --- |
| Dataset generation | both generation scripts, per split bundle | the instance's generation seed | none | dataset key from the selected dataset config, instance index, own split |
| Classifier training | both classifier scripts | resolved from the config | train and val datasets | copied from the train dataset's record |
| Pretraining, training | `fit_run` / `save_result` in `afabench.fit.run` | the contract seed | datasets, classifier, pretrained model | copied from the train dataset's record; `method_name` from the contract |
| Evaluation | `AFAEvaluator` | resolved from the config | method, eval dataset, classifier | copied from the eval dataset's record; `method_name` from the method bundle's record |
| Transformation | transform script | not applicable | not applicable | propagated, see below |
| Aggregation, plotting | nothing | | | |

Identity copied from several inputs must agree: train and val datasets with
different dataset keys or instance indices raise an error naming both
values.

ADR 0001's contract field list gains `method_name` on `TrainingContract`,
rendered by the Snakemake argument renderer from the rule's method wildcard.
_This amends ADR 0001's field list._ It is justified because a method name
is pipeline-level identity that a method bundle cannot otherwise know: one
script serves several method names (`odin_model_free` and
`odin_model_based`, `ol_with_mask` and `ol_without_mask`). The
library-not-framework decision and the `save_result` metadata shape are
unchanged. Pretrained models are shared across method names and get no
method name.

## Evaluation identity columns

The evaluator writes these constant-per-file columns into the raw table,
using the names the transformed tables already use so that merging and
plotting are unchanged:

- `afa_method`, `train_seed`, `train_hard_budget`,
  `train_soft_budget_param`: from the method bundle's record.
- `dataset`, `dataset_instance_index`, `eval_split`: from the eval dataset's
  record. `dataset` holds the dataset key.
- `initializer`: the initializer config name the evaluation selected.
- `eval_seed` (now never null), `eval_hard_budget`,
  `eval_soft_budget_param` (not written today).

ADR 0004 adds two per-episode columns, `generation_index` and
`split_index`, which unlike these vary between rows.

`SavedEvaluationSchema` gains these columns. A column whose source bundle
predates this ADR is null, not guessed. Pretraining seed, Unmasker,
classifier identity and the rest stay in the record only; they can be
recovered by following `inputs`.

The transform step then only derives plotting columns: it adds
`n_selections_performed`, melts the prediction columns into `classifier` and
`predicted_class`, and normalises nullable dtypes. It carries
`dataset_instance_index` and `eval_split` through the melt, so transformed tables
no longer lose them. The Snakemake rule keeps passing its
wildcards, but they become checks instead of sources: a value that
disagrees with a non-null column raises, and only a column that is missing
or null is filled from its argument. That keeps tables written before this
ADR transformable, including when Snakemake reruns the transform for them.
They are validated against the column set they were written with.

## Reproducibility policy

**RNG ownership.** The stage's entry point owns the seed: the
`fit_run` helper, the evaluator, the classifier scripts and the dataset
generators. It resolves a null seed once by drawing one, seeds Python,
NumPy and torch (CPU and CUDA), passes that resolved seed, never `None`, to
every component's `set_seed` and to the evaluation sampler, and records it.
`set_seed`'s return value must be used. A component may own an independent
generator only if its seed is an explicit config value recorded in its
bundle.

**Determinism.** `torch.use_deterministic_algorithms` stays off; `set_seed`
keeps cuDNN deterministic mode on and benchmark mode off. Enabling deterministic
algorithms raises on operations that have no deterministic CUDA
implementation and requires `CUBLAS_WORKSPACE_CONFIG`. The promise is: the
same seed, code, environment and device type reproduce a CPU run; CUDA
runs are reproducible on a best-effort basis. The record's `compute` and
`environment` fields say which case applies.

**Precision.** Classifier training, evaluation and most pretraining and
training paths set float32 matmul precision to `medium`, which permits
reduced-precision internal matmuls (TF32 or bfloat16) on hardware that
supports them. This stays the policy, including for evaluation. EDDI
training never sets it and runs at torch's default `highest`. That
inconsistency is kept, not fixed, because aligning it would change
results; the recorded precision makes it visible. A method whose discrete choices are sensitive to it raises the
precision locally around those matmuls, as AACO's `_exact_matmul` does.
The precision is recorded. Changing it for evaluation would change results
and would have to be documented as a result-affecting change between
releases (#36). Evaluation batch size can be result-affecting for the same
reason; it is part of the recorded resolved config.

### Observed behaviours: policy or bug

| Behaviour | Verdict |
| --- | --- |
| Deterministic algorithms not enabled | Policy, above. |
| Matmul precision `medium` by default (looser than the TF32-only `high`) | Policy, above. |
| Data loaders shuffle without an explicit generator | Policy. Every loader is single-process, so shuffling draws from the global torch generator that the entry point seeded. It is reproducible for an unchanged code path; any change in global RNG consumption reorders it, which ADR 0001's migration bar already accepts. A loader that uses worker processes must pass a generator and seed its workers. |
| AACO's `mask_seed` independent of the configured seed | Policy. Introduced deliberately so that the candidate masks are a reproducible property of the trained method bundle, the same across evaluation seeds and batch orders. It is a method hyperparameter and is already recorded through the method config. Consequence: variation across training seeds and dataset instances excludes candidate-mask variation. |
| The workflow derives pretraining, training and evaluation seeds from the dataset instance index and trains the external classifier with seed 0 | Policy, recorded as-is. |
| `set_seed` sets `PYTHONHASHSEED` at runtime | Ineffective for the running interpreter (only for child processes) but harmless. Code must not depend on the iteration order of string sets. No change. |
| Evaluation with `seed=null`: the drawn seed is discarded; initializer, Unmasker, method and sampler receive `None`; `eval_seed` is written as null | Bug. `RandomInitializer` then falls back to a fresh `torch.Generator()`, whose seed is a fixed constant, so a null seed is not random for the initializer and differs from the seed the global RNGs used. |
| Classifier seed may be null and is then recorded as null | Bug, fixed by RNG ownership. |
| `eval_soft_budget_param` is not written to the raw table | Bug, fixed by the identity columns. |
| The two dataset generators write different metadata keys | Bug, superseded by the record. |
| `docs/reference/bundle_format.md` shows `bundle_version` as integer `1`; the code writes `"1.0.0"` | Documentation bug, fixed with the manifest change. |
| AACO training reads `split_idx`, which the dataset generator never writes | Already fixed: #53 moved AACO to `save_result`, and nothing reads that key any more. |

## Required tests

These are the contract the implementation must meet. All use `tmp_path` and
generated data and run in the default suite, except end-to-end checks
marked `pipeline`.

1. **Bundle round trip.** A record with every field populated, including
   nulls, nested `resolved_config` and several inputs, saved with
   `save_bundle` and read back after `load_bundle`, equals the original.
2. **Legacy bundle.** A manifest without `provenance` or `content_hash`
   still loads, and reading its provenance reports it as absent instead of
   returning a default record.
3. **Content hash.** Identical `data/` contents give the same hash; changing
   one data file changes it; a consumer's input entry carries the hash its
   producer recorded.
4. **Capture errors.** A config value that is not JSON-serialisable, and
   train and val inputs that disagree on dataset identity, raise errors
   naming the value.
5. **Code identity.** Capture outside a git work tree records null commit
   and dirty flag; in a temporary repository, modifying a tracked file sets
   `code_dirty`.
6. **Parquet round trip.** A table saved by the evaluator yields an equal
   record from its Arrow schema metadata. `pandas.read_parquet` without
   importing `afabench` sees the identity columns with their documented
   dtypes, and the table validates against `SavedEvaluationSchema`.
7. **Resolved seed.** An evaluation with `seed=null` and a random
   initializer records the same integer in the record and in `eval_seed`,
   and re-running with that seed on CPU produces an identical table.
8. **Hand run.** Running the evaluation script outside Snakemake on smoke
   bundles produces non-null identity columns that match the inputs'
   records.
9. **Transform.** Identity flows from raw columns without arguments; a
   disagreeing argument raises; a pre-ADR raw table is filled from
   arguments; the output keeps `dataset_instance_index`, `eval_split` and
   the source record.

## Consequences

- The release manifest (#63) is assembled from these records plus
  release-level facts (release identity, coverage, workflow config), and
  does not re-derive per-artifact identity from paths. Publish and download
  tooling remain the only HF-aware code.
- Once artifacts carry identity, the remaining reasons for the long
  artifact paths become Snakemake target naming only; #47 reassesses the
  shared path builder on that basis.
- Outputs written before this change keep loading. Their records are absent
  and their identity columns null, and transform fills them from wildcards
  as today.
- Out of scope, observed during the investigation: the external classifier
  is trained only on dataset instance 0. For datasets whose instances are
  reshuffles of the same data (`accepts_seed()` is false), the test split of
  instance k overlaps instance 0's training split. The records make the
  classifier's instance index visible next to the evaluated one, but this
  ADR makes no claim about whether results are affected.
