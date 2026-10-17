# Interactive preview

The [interactive preview](../../CONTEXT.md) is a terminal browser of the
[pipeline plan](../../CONTEXT.md) of one pipeline invocation: every job of
the requested targets' dependency graph, grouped by pipeline stage, method
and dataset, with its [job status](../../CONTEXT.md) and Snakemake's reasons
for it, which can be regrouped, filtered and searched. It runs no job, submits nothing and writes no file. The
[pipeline launcher](pipeline_launcher_command.md#the-interactive-preview) opens
it with `--interactive-preview`; steps:
[run the pipeline](../how-to/maintainer/run_the_pipeline.md#inspect-planned-work).

## Statuses

Each job wears its status as a text badge with a color of its own, and
each group counts its jobs by status. Both statuses are shown; each job
appears and counts once, a shared prerequisite included.

| Status | Badge | Meaning |
| --- | --- | --- |
| Run | `Run`, green | The invocation schedules the job, including to replace outputs that exist but are stale. |
| Reuse | `Reuse`, blue | The invocation uses the job's existing outputs without running it. |

The statuses and reasons are Snakemake's own need-run decisions for the
invocation, as a real run would make them: the preview forces no job and
reads no file existence itself. A plan with nothing to run is a valid
preview that still shows the jobs it reuses. A file an input needs that no
rule of the invocation produces, such as the output of a stage that is off,
is no job of the plan: a job reading it lists it under `Existing inputs`.

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

## Grouping, filters and search

The controls above the tree regroup, filter and search the jobs of the plan
already loaded: nothing is planned again, and the invocation, its targets
and configuration stay as they are. Settings last until the preview quits.

| Control | Effect |
| --- | --- |
| Group by | The dimensions of the tree's levels, in order: `stage → method → dataset` (the default, or the launcher's [`--group-by`](pipeline_launcher_command.md#arguments)), any other order of the three, or any one or two of them, such as `dataset → method`. A dimension left out is still in each job's details. |
| Stage, dataset and status filters | Show only the jobs of the chosen pipeline stage, dataset or job status; `All` shows every one. They offer the values and statuses the plan's jobs have. Strict: a job without the chosen value, such as an aggregation job under a dataset filter, is hidden. |
| Method filter | Show the chosen method's jobs and the jobs they need, directly or not, such as their shared prerequisites, and no other shared prerequisite. The other filters still apply to the jobs it keeps. |
| Search (`/`) | Show only the jobs whose rule, stage, method, dataset and tree label contain every whitespace-separated term, regardless of case, with the matching groups expanded. |

Filters and search compose. Groups count the jobs shown. When some jobs are
hidden, the status line counts the shown jobs beside the whole plan's, as
`Shown: 3 of 9 jobs: 1 Run, 2 Reuse | Whole plan: 9 jobs: 5 Run, 4 Reuse`;
when none match, the tree says so. A job hidden by a filter is still named,
with its status, among the jobs another job needs or is needed by.

## Panes

| Pane | Contents |
| --- | --- |
| Controls | The grouping, the stage, method, dataset and status filters, and the search field. |
| Status line | The planning state, then the number of jobs and their counts by status, and when some are hidden, the shown jobs' counts beside the whole plan's. |
| Jobs | The jobs as a tree, grouped as the controls choose, each group labelled with its counts by status. Stages come in pipeline order. Where a method level applies, a job whose job record names no method, such as a dataset generation, classifier training or pretraining job, is under `Shared prerequisites`, and a job that writes no job record, such as an aggregation or visualization job, under `Across methods`; where a dataset level applies, such a job is under `Across datasets`. Each job is in one place and counts once. Jobs below the last level are labelled by their status badge, rule, dataset realization, seeds and training and evaluation budgets. |
| Details | For a group: its dimension value, its jobs by status and its subgroups' counts. For a job: its status and the reasons for it, rule, stage, method, dataset, dataset realization, seeds, training and evaluation hard budgets and soft-budget parameters, evaluation batch size, wildcards, device, CPUs, GPUs, requested time limit, input and output paths, existing inputs no job of the plan makes, the jobs it needs and the jobs that need it, each with its status, and for a shared prerequisite the methods that consume it. A dimension or identity field the job has none of reads `not applicable`. |

The reasons are Snakemake's: missing outputs, inputs newer than the job's
outputs, inputs another job of the plan rewrites, a changed set of inputs,
code, params or software environment, a forced run, or for a reused job that
the outputs it is needed for exist and are up to date and nothing it needs
is rerun.

Both panes are side by side in a terminal of 80 columns or more. In a
narrower terminal the controls take two columns and the focused pane fills
the terminal: `Tab` and `Enter` switch to the details, `Tab` and `Escape`
back to the tree.

## Keys

| Key | Action |
| --- | --- |
| `↑`, `↓` | Move through the tree, or scroll the details. |
| `→` | Expand the selected group. |
| `←` | Collapse the selected group, or move to its parent. |
| `Enter` | Show the selected group's summary or job's details. |
| `Tab` | Switch between the tree and the details. |
| `Escape` | Back to the tree. |
| `g` | Go to the grouping control. |
| `f` | Go to the filters, starting at the stage filter. |
| `←`, `→` | On a control, move to the previous or next one. |
| `Enter`, `↓` | On a control, open its options; `Enter` chooses one. |
| `/` | Search: typing narrows the tree; `Enter` or `Escape` goes back to it. |
| `c` | After a planning failure, copy the diagnostics to the clipboard. |
| `q` | Quit: exit status 1 after a planning failure, else 0. |

The footer lists the keys, and the tree's border the arrow keys.
