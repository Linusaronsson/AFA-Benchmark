This directory contains reusable hydra components that can be used in multiple different scenarios.

## Layout

`extra/conf` has three tiers:

- `components/` holds configs shared between scripts, as one flat set of
  groups.
- `scripts/<script_group>/<script_name>/` holds the configs of a single
  script.
- `global/` and `release/` hold Hydra and release settings.

Every script config lists `file://extra/conf/components` in
`hydra.searchpath`, so each group directory in `components/` is also a
top-level group. That is how scripts select `initializer=<name>`,
`unmasker=<name>` and `dataset_key=<key>` (the plain training contract of
`docs/adr/0001-training-contract-as-library.md`) while the files stay here.

| Group | Used by |
| --- | --- |
| `initializer`, `unmasker`, `dataset_key` | pretraining, training, classifier and evaluation scripts |
| `masked_pretraining`, `masking_probabilities` | `pretrain_model` of JAFA, ODIN and OL |
| `mdps`, `rl_training_loops` | `train_method` of JAFA, ODIN and OL |

- `components/dataset_key/<key>.yaml` sets the `dataset_key` config value.
  Every dataset key in `extra/workflow/conf/datasets/` needs one.
- Each method's root config ends with
  `optional experiment@_global_: ${dataset_key}`, which loads the method's
  `experiment/<key>.yaml` when it exists.

## Hydra gotchas

All were verified against the Hydra version in `uv.lock`.

- **A group override must use the exact key the defaults list declares.** A
  config with `- /components/initializer@initializer: ???` would only accept
  `components/initializer@initializer=...`. Select the group by its search-path
  name (`- initializer: ???`) so that `initializer=cold` works. The group
  directory name is the CLI key, so it is singular.
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
