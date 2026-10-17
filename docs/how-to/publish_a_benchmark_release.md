# Publish a benchmark release

For a [maintainer](../explanation/user_types.md#maintainer): publish benchmark
outputs on Hugging Face. For background, see
[benchmark releases](../explanation/benchmark_releases.md).

## 1. Set up the host (once)

1. Create a Hugging Face [user access token](https://huggingface.co/settings/tokens)
   with the `write` role. Do not commit it.
2. Authenticate on the machine you will publish from:

   ```shell
   uv run hf auth login
   uv run hf auth whoami
   ```

   For non-interactive use, set `HF_TOKEN` instead.
3. Create a public dataset repository:

   ```shell
   uv run hf repo create afabench/results --repo-type dataset
   ```

   Replace `afabench/results` throughout this guide if publishing elsewhere.
   Use the full `<username>/<repo>` or `<organization>/<repo>` id, with the
   exact owner slug shown by `hf auth whoami`. For an organization, confirm
   that your account has permission to create and write repositories there.

If authentication fails, run `uv run hf auth whoami`, check any `HF_TOKEN`
value, and log in again with a valid token.

## 2. Save the release package

1. [Run the benchmark pipeline](run_the_pipeline.md) and wait for it to
   finish. Use an output root containing only the outputs you intend to
   publish.
2. Check the datasets' entries in `conf/release/dataset_redistribution.yaml`
   against the intended
   [dataset release policy](../reference/release_manifest.md#dataset-redistribution).
3. Choose a new release id and a fresh package directory outside the
   repository. Use letters, digits, `.`, `_` and `-` for the id, starting
   with a letter or digit. Replace `example` below with your choices:

   ```shell
   uv run python scripts/release/snapshot.py save ~/data/afabench-releases/example \
       --release-id example
   ```

Do not repeat the pipeline's configuration arguments; `save` reads the
recorded run configuration. Nothing is uploaded yet. See the
[`save` reference](../reference/snapshot_command.md#save) for other options.

Inspect the generated `README.md` in the package directory to check its
coverage summary and output links. To include a description and caveats,
write a Markdown file and pass `--release-notes /path/to/notes.md` when saving.
For example: “Initial benchmark results for the tabular datasets.”
The description is optional and appears in the release's README on Hugging
Face, not on a separate page in this repository.

Leave the saved package unchanged, including while publishing.

## 3. Publish

```shell
uv run python scripts/release/snapshot.py publish ~/data/afabench-releases/example \
    --repo-id afabench/results
```

`publish` validates the package automatically. Open the release web address
printed on success.

If validation fails, fix the reported problem in the source output root and
save to a fresh package directory before retrying. Restore missing inputs or
regenerate incomplete evaluations as directed; do not edit the manifest to
bypass a failure. For dirty or unknown producing code, regenerate from a
clean commit, or explicitly pass `--allow-dirty-code` if intentional. See
[publication checks and overrides](../reference/snapshot_command.md#publish).

To correct an already published release, use a new release id. You can note
what it corrects in the new release's README by supplying `--release-notes`
when saving.
