This directory contains reusable hydra components that can be used in multiple different scenarios.

## Hydra gotchas

Both were verified against the Hydra version in `uv.lock`.

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
