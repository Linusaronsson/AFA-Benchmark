# Publishing and downloading benchmark releases

A **benchmark release** is an output snapshot saved with a release manifest
(see [`release_manifest.md`](release_manifest.md) and
[`tutorials/output_snapshots.md`](tutorials/output_snapshots.md)) and
published on Hugging Face. Two commands of `scripts/release/snapshot.py`
handle the host, and nothing else in AFABench does: train, evaluation and
plotting scripts read and write local files only and never need a Hugging
Face account, token or service. Design background:
[`artifact_publishing.md`](artifact_publishing.md).

- `publish` uploads a prepared snapshot. Only a maintainer running it
  publishes anything; finishing a pipeline run or saving a snapshot does
  not.
- `download` fetches one release, the latest full release unless a release
  id is given, and restores all of it (`--all`) or only the selected
  payloads, as the offline `restore` would.

## Where releases live

Releases are stored in one Hugging Face **dataset** repository, given with
`--repo-id` or the `AFABENCH_RELEASE_REPO` environment variable. Each
release is a folder holding the snapshot exactly as `save` wrote it:

```text
releases/<release_id>/
    release_manifest.json
    output_mtimes.json
    output/
        eval_results/...              raw evaluation tables
        eval_results_transformed/...  plotting-ready tables
        merged_results/...
        plot_results/...
        ...                           every other file the snapshot holds
test_releases/<release_id>/           test-only packages, same layout
```

A published release is never replaced: publishing a release id that already
exists is refused, so each id always names the same outputs. Publish a
corrected release under a new id. `output_mtimes.json` records the mtimes
of everything under `output/`, which Hugging Face does not keep, so that a
download restores them for Snakemake (see "Why mtimes are preserved" in the
snapshot tutorial).

## Finding and downloading results without AFABench

Public releases need no account and no AFABench install. Browse
`https://huggingface.co/datasets/<repo_id>/tree/main/releases` to find a
release, read its `release_manifest.json` for the producing commit,
configuration, coverage and the identity of each table, and download single
files from

```text
https://huggingface.co/datasets/<repo_id>/resolve/main/releases/<release_id>/output/<path>
```

with a browser, `curl`, or any Parquet reader that accepts URLs:

```python
import pandas as pd

df = pd.read_parquet(
    "https://huggingface.co/datasets/<repo_id>/resolve/main/releases/"
    "<release_id>/output/eval_results_transformed/<table path>/eval_data.parquet"
)
```

Tables are stored exactly as the pipeline wrote them. For the difference
between raw and plotting-ready tables, and for the dataset instance and
evaluation split that transformed tables do not hold in their columns, see
[`release_manifest.md`](release_manifest.md#raw-versus-plotting-ready-tables).

## Downloading a release into a checkout

```shell
uv run python scripts/release/snapshot.py download --all --repo-id <repo_id>
```

Downloads every file of the **latest full release** and restores its
outputs into `extra/output` (override with `--destination-root`) and its
manifest into `extra/release_manifest.json`. It then prints the release's
scope, execution mode, commit and workflow configuration. No token is
needed for a public repository.

### Choosing the release

- **Latest (default).** The release argument defaults to `latest`: the
  `full` release whose manifest `created_at` is newest. A newer `partial`
  release is never chosen this way. If no full release is published, the
  command stops and lists the published releases with their scopes; it
  does not fall back to a partial one. `latest` is therefore never a
  release id. It cannot name a test release either.
- **Pinned.** Name a release id to download that release, whether older or
  partial:

  ```shell
  uv run python scripts/release/snapshot.py download 2026-04-kdd26 --all \
      --repo-id <repo_id>
  ```

The release is resolved once, before anything is downloaded. Everything
the command fetches comes from that one release, even if a newer release is
published while it runs. It never assembles outputs from several releases.

### Selecting what to download

Without `--all`, name what to fetch, by payload category
([`release_manifest.md`](release_manifest.md#payload-categories)) and output
category, and optionally narrow it by coverage. Giving neither, or `--all`
together with a selection, is an error.

| Option (repeatable) | Selects |
| --- | --- |
| `--payload-category` | `raw_evaluation_table`, `transformed_evaluation_table`, `dataset_bundle`, `classifier_bundle`, `pretrained_model_bundle`, `afa_method_bundle`. |
| `--output-category` | A top-level folder of the output root listed in the manifest's `coverage.output_categories`, such as `plot_results` or `merged_results`, fetched whole. Coverage options do not narrow it, since the manifest does not describe its files. |
| `--dataset`, `--method`, `--dataset-instance`, `--eval-split`, `--initializer` | Evaluations with that dataset key, method, dataset instance index, evaluation split or initializer. |
| `--budget-setting` | `hard_budget` or `soft_budget` evaluations; the two are never merged. |
| `--classifier-variant` | Evaluations whose raw table holds `builtin` or `external` predictions. |

Coverage options select entries of the manifest's `evaluation_tables`.
Options of one kind match any of their values; options of different kinds
must all match; a kind not given does not restrict. The payload categories
then decide what of those evaluations is fetched:

- the evaluation tables of the selected evaluations, raw or transformed;
- the bundles those evaluations were produced from, found by following
  the `inputs` of the tables and, in turn, of the bundles, and kept only if
  their category is selected. Following inputs includes what a needed
  bundle was trained from. For example, the external classifier is trained
  on dataset instance 0, so selecting it also selects that instance's
  `train` and `val` dataset bundles. Without them the workflow would
  regenerate those splits and retrain the classifier.
- with each pretrained-model and AFA-method bundle, the time record its
  pretraining or training job wrote beside it (`pretrain_time.txt`,
  `train_time.txt`): the folder holding that one job's outputs is fetched.
  The workflow's time aggregation reads these records, so without them it
  would rerun the job and replace the restored bundle.

Bundles are never fetched unless their category is named. So a
results-only download fetches no bundle, and naming `dataset_bundle` and
`classifier_bundle` gives the shared prerequisites of the selected
evaluations without any AFA-method bundle. Method-specific classifiers
(`method_name` set) come only with the evaluations of their method.

Plotting baselines for comparison, with plots, and no bundles:

```shell
uv run python scripts/release/snapshot.py download --repo-id <repo_id> \
    --payload-category transformed_evaluation_table \
    --payload-category raw_evaluation_table \
    --output-category plot_results \
    --dataset cube --dataset physionet --budget-setting hard_budget
```

Shared prerequisites for training and evaluating a new method on the same
inputs, from a pinned release:

```shell
uv run python scripts/release/snapshot.py download 2026-04-kdd26 \
    --repo-id <repo_id> \
    --payload-category dataset_bundle --payload-category classifier_bundle \
    --dataset cube
```

Add `--payload-category afa_method_bundle` (and `pretrained_model_bundle`)
with `--method <name>` to fetch the bundles needed to re-evaluate a
published method.

To add your own method to the published comparison without reproducing
the baselines, follow
[`tutorials/compare_with_published_baselines.md`](tutorials/compare_with_published_baselines.md).

### Coverage reports

Requested coverage that the release lacks is reported after the download.
Examples are a dataset, method or setting that no evaluation matches, or a
selected table, bundle or output category that the release does not hold:

```text
Downloaded 6 file(s) and 1 folder(s).
Missing from release 2026-11-partial:
  dataset 'physionet': no evaluation of the release matches
  transformed_evaluation_table eval_results_transformed/...: not in the release
```

What is present is still downloaded. If nothing selected is present, the
command fails, writes nothing and prints the same report. Nothing is taken
from another release and no training or evaluation is run. Running the
pipeline afterwards produces genuinely missing outputs.

### Existing files and provenance

The overwrite rule of `restore` applies to both whole and selective
downloads. If any file the download would write already exists, including
`release_manifest.json` beside the destination root, nothing is written and
the conflicting paths are listed. `--overwrite` replaces only those files.
The restored `release_manifest.json` is the release's complete manifest,
whatever was selected. As a result, a second download into the same
destination needs `--overwrite` or another destination. An unknown release
id, or one that names a test release (see below), is an error.

The command does not check that your checkout is compatible with the
release's commit, and it permits any release regardless. Read the printed
provenance (commit, dirty flag, workflow profile, config files and
overrides) and decide whether the outputs suit your code; see
[`artifact_publishing.md`](artifact_publishing.md#release-contract).

## Publishing a release

1. **Create the repository once.** A Hugging Face dataset repository, public
   for an announced release:
   `hf repo create <repo_id> --repo-type dataset`. `publish` does not create
   it.
2. **Authenticate.** Only publishing needs a token with write access, from
   `hf auth login` or the `HF_TOKEN` environment variable. Nothing else in
   AFABench reads it.
3. **Prepare the package** from the run's outputs with a release manifest:

   ```shell
   uv run python scripts/release/snapshot.py save /path/to/2026-10-kdd26 \
       --release-id 2026-10-kdd26 --scope full \
       --profile extra/workflow/profiles/config/kdd26
   ```

   Release ids are letters, digits, `.`, `_` and `-`.
4. **Review the package** (next section). Nothing has been uploaded yet.
5. **Publish it:**

   ```shell
   uv run python scripts/release/snapshot.py publish /path/to/2026-10-kdd26 \
       --repo-id <repo_id>
   ```

   The whole release is uploaded in one commit, and the command prints the
   release's provenance and its web address.

`publish` refuses a snapshot without a release manifest, a manifest version
this checkout does not read, a release id already published, and a
`test_only` package.

### Dataset redistribution

`publish` also refuses an official release while any dataset key in the
manifest's `settings.dataset_redistribution` is `unreviewed` or
`restricted`, and names each of them. Record the review in
`extra/conf/release/dataset_redistribution.yaml` and save the package again
([`release_manifest.md`](release_manifest.md#dataset-redistribution)), or
remove the dataset from the run. To publish such a dataset anyway, allow it
by its dataset key; the option repeats, once per dataset:

```shell
uv run python scripts/release/snapshot.py publish /path/to/2026-10-kdd26 \
    --repo-id <repo_id> --allow-redistribution cube
```

There is no option that allows every dataset at once. Allowing a dataset
that is not unreviewed or restricted in the manifest is an error. The
published manifest keeps the dataset's status, so it stays visible to
everyone who reads the release, and the host commit names the allowed
datasets. Test releases are not checked and take no allowance, since they
are never public benchmark releases.

### Test releases

Smoke-test outputs are always `test_only` and never become official
releases. To check the real host round trip before announcing a release,
publish a smoke package as a **test release**:

```shell
uv run python scripts/release/snapshot.py publish /path/to/smoke-check \
    --repo-id <repo_id> --test-release
uv run python scripts/release/snapshot.py download smoke-check --all \
    --repo-id <repo_id> --test-release --destination-root /tmp/check/output
```

Test releases live under `test_releases/`, apart from official releases.
`--test-release` accepts only `test_only` packages, and a test release can
only be downloaded with `--test-release`, so it cannot be fetched or
announced as an official release. Name the test release by its id; the
selection options work as for official releases. Prefer a separate scratch
repository for such checks.

## Maintainer review checks

Before publishing, read the package's `release_manifest.json` and check:

- **Completeness.** `coverage` lists the datasets, dataset instances,
  methods, evaluation splits, budget settings, classifier variants and
  output categories actually present. Every `evaluation_tables` entry the
  release is meant to cover has `raw_present` and `transformed_present`.
  `scope` is `full` only if the release covers the whole benchmark
  configuration; otherwise it is `partial`. A `full` release becomes the
  default download for everyone as soon as it is the newest full one. The package's `output/` holds
  no stale or unrelated files, since the snapshot copies the output root
  verbatim.
- **Execution mode.** `execution_mode` is `production`. Smoke outputs are
  `test_only` and cannot be published as an official release, but check
  that `workflow_config` matches the intended run and that `code.dirty` is
  false, so `code.commit` describes the code that ran.
- **Permission to redistribute.** Every dataset, dataset bundle and other
  payload in `output/` may be redistributed publicly under its licence.
  Generating a bundle does not grant that right. Every dataset key in
  `settings.dataset_redistribution` must be `permitted`; `save` prints the
  unreviewed and restricted ones, and `publish` refuses them unless each is
  allowed by name (see [Dataset redistribution](#dataset-redistribution)).
  Review, remove, or publish elsewhere anything whose permission is
  unresolved before publishing
  ([`release_manifest.md`](release_manifest.md#dataset-redistribution)).
  `coverage.payloads` and the `bundles` entries show what each payload
  category holds and its size; production sizes are still unknown until
  `snapshot.py inventory` is run on the real outputs.

Also record the changes since the previous release in the release notes
(next section).

## Recording result-affecting changes between releases

Users compare their own results, produced with their checkout, against a
release produced with another commit. Two questions decide whether that is
valid, and a release answers both separately:

- **File/API compatibility**: whether the release's files still load and
  fit the checkout's pipeline. Examples are manifest and bundle versions,
  table schemas, output paths and registered class names. A mismatch shows
  up as an error, such as an unknown `manifest_version`, a bundle class
  missing from the registry, or a `MissingInputException` for a table at
  an unexpected path.
- **Scientific comparability**: whether results produced now would have
  been produced the same way as the release's. Many changes break it
  silently, with every file still loading:
  - dataset generation or preprocessing, and dataset instances or splits;
  - feature costs;
  - Unmaskers, initializers and acquisition semantics, such as stop
    handling, forced acquisition or budget accounting;
  - the external classifier's architecture or training;
  - method defaults and hyperparameters;
  - evaluation batch sizes, seeds and metrics.

Compatibility of a checkout is not checked automatically. Maintainers
record both kinds of change in [`release_notes.md`](release_notes.md), one
entry per release, newest first, written while reviewing the package and
before publishing it. Each entry has:

- the release id, scope and producing commit;
- the previous release it is compared with, if any;
- **compatibility changes**: format, schema, path or API changes since that
  release, and what an older checkout or release needs to work with them;
- **result-affecting changes**: every change to data, splits,
  preprocessing, feature costs, acquisition semantics, classifiers, method
  configuration or metrics since that release. Name the commits and the
  affected datasets, methods and settings, and say whether results of the
  previous release remain comparable with this one. "None" is an entry
  too: say so explicitly.

To find candidates, review `git log <previous commit>..<new commit>` and
the difference between the two manifests' `workflow_config.merged` and
`settings`. Pay attention to `extra/conf/`, `extra/workflow/`,
`afabench/` and `scripts/`. A corrected release is published under a new
id, with an entry saying what it corrects.

## Limits of this version

- "Latest" is decided by the manifests' `created_at`. Finding it reads the
  manifest of every published release.
- Output categories other than payload categories (`plot_results`,
  `merged_results`, ...) are fetched whole, not by dataset or method.
- The release contents are whatever the snapshot holds; the manifest
  lists its native bundles and payload categories
  ([`release_manifest.md`](release_manifest.md#payload-categories)).
  `publish` refuses a release with an unreviewed or restricted dataset
  rather than leaving that dataset's payloads out, and does not exclude
  payloads by category.
- Empty directories are restored from `output_mtimes.json`; nothing else
  about a file but its bytes and mtime is kept.
