# Interactive preview

The [interactive preview](../../CONTEXT.md) is a terminal browser of the
[pipeline plan](../../CONTEXT.md) of one pipeline invocation: the jobs the
invocation would schedule, grouped by pipeline stage, method and dataset. It
runs no job, submits nothing and writes no file. The
[pipeline launcher](pipeline_launcher_command.md#the-interactive-preview) opens
it with `--interactive-preview`; steps:
[run the pipeline](../how-to/maintainer/run_the_pipeline.md#inspect-planned-work).

It shows scheduled work only. Jobs whose existing outputs the invocation
reuses are not shown, and the preview says so in its status line.

## Planning

The preview opens at once with a planning indicator and plans the invocation
in a child process, `python -m afabench.pipeline_plan`, through the shared
planner that the [compute estimate](estimate_compute_command.md) uses, so both
describe the same scheduled jobs. A config invocation of a downloaded release
is planned as its release preview, as a dry run is. Quitting while planning
ends the child process.

When planning fails, as it does for an invocation Snakemake rejects, the
preview shows a failure state and the child's output in a scrollable text
area, where text can be selected and `c` copies it all to the clipboard. No
partial plan is shown. Quitting then exits with status 1.

## Panes

| Pane | Contents |
| --- | --- |
| Status line | The planning state, then the number of scheduled jobs. |
| Jobs | The scheduled jobs as a tree: pipeline stage → method → dataset → job, each group labelled with its job count. Stages come in pipeline order. A job whose job record names no method, such as a classifier training job, is under `no method`; a job that writes no job record, such as an aggregation or visualization job, is under `no method` and `no dataset`. Jobs below the last level are labelled by their rule, dataset realization, seeds and training and evaluation budgets. |
| Details | For a group: its dimension value, its number of scheduled jobs and its subgroups' counts. For a job: its rule, stage, method, dataset, dataset realization, seeds, training and evaluation hard budgets and soft-budget parameters, evaluation batch size, wildcards, device, CPUs, GPUs, requested time limit, and the scheduled jobs it needs. A dimension or identity field the job has none of reads `not applicable`. |

Both panes are side by side in a terminal of 80 columns or more. In a
narrower terminal the focused pane fills the terminal: `Tab` and `Enter`
switch to the details, `Tab` and `Escape` back to the tree.

## Keys

| Key | Action |
| --- | --- |
| `↑`, `↓` | Move through the tree, or scroll the details. |
| `→` | Expand the selected group. |
| `←` | Collapse the selected group, or move to its parent. |
| `Enter` | Show the selected group's summary or job's details. |
| `Tab` | Switch between the tree and the details. |
| `Escape` | Back to the tree. |
| `c` | After a planning failure, copy the diagnostics to the clipboard. |
| `q` | Quit: exit status 1 after a planning failure, else 0. |

The footer lists the keys, and the tree's border the arrow keys.
