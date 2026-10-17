# Publish a benchmark release

For a [maintainer](../../explanation/user_types.md#maintainer):
publish benchmark outputs on Hugging Face.

## 1. Set up the host

1. Create a [Hugging Face token](https://huggingface.co/settings/tokens) with
   the `write` role. Do not commit it.
2. Authenticate on the publishing machine:

   ```shell
   uv run hf auth login
   uv run hf auth whoami
   ```

   For non-interactive use, set `HF_TOKEN` instead.
3. Create a public dataset repository if it does not exist:

   ```shell
   uv run hf repo create afabench/results --repo-type dataset
   ```

   When publishing elsewhere, replace `afabench/results` throughout with
   your exact `<username>/<repo>` or `<organization>/<repo>` id. Verify write access.

If authentication fails, check `HF_TOKEN`, run `uv run hf auth whoami` and log in again.

## 2. Save a release package

1. [Run the pipeline](run_the_pipeline.md) to completion. Keep only intended
   outputs in the output root.
2. Review `conf/release/dataset_redistribution.yaml` against the
   [dataset release policy](../../reference/release_manifest.md#dataset-redistribution).
3. Choose an unused release id and a fresh destination outside the repository.
   Use letters, digits, `.`, `_` and `-`, starting with a letter or digit:

   ```shell
   uv run python scripts/release/snapshot.py save ~/data/afabench-releases/example \
       --release-id example
   ```

   Do not repeat pipeline configuration arguments. For release notes, add
   `--release-notes /path/to/notes.md` when saving.
4. Check the generated README's coverage, notes and output links.
   Leave the package unchanged after saving, including during publication.

## 3. Publish

```shell
uv run python scripts/release/snapshot.py publish ~/data/afabench-releases/example \
    --repo-id afabench/results
```

Open the release URL printed on success.

If validation fails, fix the source outputs and save to a fresh destination
before retrying. Do not edit the manifest to bypass validation. For dirty or
unknown producing code, regenerate from a clean commit or explicitly use
`--allow-dirty-code` if intentional.
[Validation and overrides](../../reference/snapshot_command.md#publish).

To correct a published release, use a new release id and describe the
correction in its release notes.
