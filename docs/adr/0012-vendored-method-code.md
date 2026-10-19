---
status: accepted
---

# Third-party method code is vendored close to verbatim, outside the repository's standards

The first L2M variants (`l2m_real_feature_prior`,
`l2m_synthetic_feature_prior`) were written from the paper alone, because
the authors' repository had no license. Where the paper was silent, the
port had to guess: the labelling function's ranges, the missingness
mechanism, the normalization, the attention layout, the fit loops. Once
the authors added an MIT license, `l2m_official` could run their code
itself.

We decided to copy that code into
`src/afabench/components/methods/discriminative/l2m/upstream/`
with as few changes as possible, instead of porting it to the
repository's style:

- **Pinned and attributed.** The folder holds the upstream license and a
  README naming the commit and the upstream path of every file.
- **Every change is marked.** Each patch carries a `# PATCH(afabench)`
  comment and is listed in the README. A patch may only change the
  environment (device handling, import paths, removed dependencies),
  storage (in-memory validation tasks), or reproducibility (seeding). It
  must not change what is computed. Unused code whose imports are not
  dependencies of this repository is removed rather than patched.
- **Exempt from the standards.** The folder is excluded from ruff and
  basedpyright (`ruff.toml`, synced to `pyrightconfig.json`), and
  `CODING_STANDARDS.md` does not apply to it.
- **Adapted at the boundary.** The fit contract, bundles, the
  `AFAMethod` protocol and anything upstream gets from libraries this
  repository lacks (`accelerate`, `transformers`) live in an adapter
  beside the folder, written to the repository's standards.

## Considered options

- **Port in the repository's style.** Rejected: a rewrite cannot be
  checked against upstream line by line, and it reopens the gap the first
  port had. Every renamed variable and restructured loop is another place
  where results could diverge from the authors' code.
- **Depend on the upstream repository as a package.** Rejected: it is not
  packaged, hardcodes CUDA, and pulls in dependencies the benchmark does not
  need (`accelerate`, `transformers`, `gpytorch`, `polars`).

## Consequences

- Diffing the folder against the pinned commit shows exactly the listed
  patches. Updating to a newer commit means copying again and reapplying
  them.
- Type checking does not report errors inside the folder, so the
  adapter's tests carry the burden of catching interface mismatches.
