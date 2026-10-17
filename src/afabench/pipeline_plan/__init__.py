"""
Plan a pipeline invocation: the jobs its Snakemake graph schedules.

`python -m afabench.pipeline_plan --output <plan.json> -- <Snakemake
arguments>` writes the plan as JSON (`plan_json`), for another process such
as the interactive preview to read.
"""
