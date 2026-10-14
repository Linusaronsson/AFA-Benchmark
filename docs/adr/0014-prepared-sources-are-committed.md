---
status: accepted
---

# A real-world dataset's prepared source is committed unless its license forbids it

Dataset generation read some sources from files nobody could rebuild, such
as a PhysioNet Challenge 2012 aggregate made by a notebook that no longer
existed, and downloaded others on first use from servers that may be slow
(PhysioNet serves the Challenge's raw files in about 15 minutes) or gone.
The two steps were mixed up: what turns a provider's raw files into the
file a dataset reads ran either nowhere in the repository or on every
generation.

We decided to separate **source preparation** from dataset generation (see
`CONTEXT.md`). A tabular dataset class owns both: `prepare_source` downloads
the provider's raw files and writes the **prepared source**, which
`scripts/dataset_generation/prepare_source.py` runs, and generation reads
the prepared source and checks it against the SHA-256 the class pins. The
prepared source is committed to the repository, in `data/<dataset key>/`
with a `LICENSE.md` giving its license and attribution, so generation never
depends on an external server. A dataset whose license does not permit
redistribution, or whose license its provider leaves unclear, is the
exception: its source is gitignored and generation prepares it when it is
missing.

## Considered options

- **Download every source at generation.** Rejected: every checkout then
  depends on each provider staying online and fast, and a source whose
  preparation is slow makes every first generation slow.
- **Commit only the sources that are slow to download.** Rejected: which
  sources count as slow changes with the providers, and the rule would have
  to be judged per dataset.
- **Commit the provider's raw files and prepare at generation.** Rejected:
  PhysioNet's raw files are 12,000 records whose aggregation takes longer
  than generation should, and a prepared file is what reviewers of a
  dataset need to read.
- **Git LFS for the committed files.** Rejected: every user would need the
  LFS tool and its server, which is the dependence this decision removes.

## Consequences

- Preparing a source again must reproduce the committed file byte for
  byte; a changed provider file fails the pinned SHA-256 and becomes a new
  dataset key rather than a silent change.
- The repository grows by every committed source, about 130 MB in plain
  git. Sources rarely change, so history grows only when one is re-prepared.
- A source the repository holds but cannot rebuild, such as diabetes, which
  is copied from the GDFS repository, says so on its dataset page.
