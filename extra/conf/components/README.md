This directory contains reusable hydra components that can be used in multiple different scenarios.

## Training contract groups

Pretraining and training scripts receive the plain training contract
(`docs/adr/0001-training-contract-as-library.md`): `initializer=<name>`,
`unmasker=<name>` and `dataset_key=<key>` instead of Hydra group paths. The
top-level groups next to this directory implement that:

- `extra/conf/initializer/<name>.yaml` and `extra/conf/unmasker/<name>.yaml`
  point at `components/initializers/<name>.yaml` and
  `components/unmaskers/<name>.yaml` with `@_here_`, so the definitions stay
  shared with the evaluation and classifier scripts, which still select
  `components/initializers@initializer` and `components/unmaskers@unmasker`.
- `extra/conf/dataset_key/<key>.yaml` sets the `dataset_key` config value.
  Every dataset key in `extra/workflow/conf/datasets/` needs one.
- Each method's root config ends with
  `optional experiment@_global_: ${dataset_key}`, which loads the method's
  `experiment/<key>.yaml` when it exists.

## Hydra gotchas

All were verified against the Hydra version in `uv.lock`.

- **A group override must use the exact key the defaults list declares.** A
  config with `- /components/initializers@initializer: ???` only accepts
  `components/initializers@initializer=cold`; `initializer=cold` fails with
  "You must specify 'components/initializers@initializer'". To accept the
  short form, make `initializer` a top-level group
  directory instead of packaging another group into it.
- **A defaults-list interpolation only sees other defaults-list choices, not
  config values.** `- optional experiment@_global_: ${dataset_key}` fails
  when `dataset_key` is a plain value set on the command line. It works when
  `dataset_key` is itself a config group (`dataset_key/<key>.yaml` with
  `# @package _global_` and `dataset_key: <key>`) selected earlier in the
  defaults list.
- **A config reached through a defaults-list interpolation may not
  `override` a group.** Hydra rejects `- override /components/x@y: z` inside
  `experiment/<key>.yaml` ("Default List Overrides are not allowed in the
  subtree of an interpolated config group"). An experiment file may include a
  group without the keyword (`- /components/x@y: z`) as long as the root
  config does not select that group itself, otherwise Hydra reports
  "Multiple values". This is why the root configs of DIME, GDFS and CAE do
  not select an architecture, and why the default masking probabilities live
  in `components/masked_pretraining/default.yaml` rather than in a group.
