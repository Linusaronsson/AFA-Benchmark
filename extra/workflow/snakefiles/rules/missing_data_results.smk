# Final artifacts for the induced study; included by missing_data.smk.

rule exact_control:
    input:
        "scripts/paper/exact_study.py",
    output:
        f"{RESULTS}/exact_study.csv",
        f"{RESULTS}/exact_study_values.csv",
    threads: 4
    shell:
        "python {input} --jobs {threads} --output-dir {RESULTS}"


rule plot_exact_control:
    input:
        data=f"{RESULTS}/exact_study.csv",
        script="scripts/paper/plot_exact_study.py",
        style="afabench/plotting/methods.py",
    output:
        f"{RESULTS}/exact_study_raw.pdf",
    shell:
        "python {input.script} --input {input.data} "
        "--output-stem {RESULTS}/exact_study_raw"


rule budget_control:
    input:
        script="scripts/paper/budget_study.py",
        problem="scripts/paper/budget_problem.py",
        reference="scripts/paper/exact_study.py",
    output:
        f"{RESULTS}/budget_study.csv",
    threads: 4
    shell:
        "python {input.script} --jobs {threads} --output-dir {RESULTS}"


rule plot_budget_control:
    input:
        data=f"{RESULTS}/budget_study.csv",
        script="scripts/paper/plot_exact_study.py",
        style="afabench/plotting/methods.py",
    output:
        f"{RESULTS}/budget_study.pdf",
    shell:
        "python {input.script} --vary-budget --input {input.data} "
        "--output-stem {RESULTS}/budget_study"


rule plot_combined_control:
    input:
        dimension=f"{RESULTS}/exact_study.csv",
        budget=f"{RESULTS}/budget_study.csv",
        script="scripts/paper/plot_exact_study.py",
        style="afabench/plotting/methods.py",
    output:
        f"{RESULTS}/exact_study_combined.pdf",
    shell:
        "python {input.script} --input {input.dimension} "
        "--budget-input {input.budget} "
        "--output-stem {RESULTS}/exact_study_combined"


rule conceptual_constants:
    input:
        "scripts/paper/conceptual_constants.py",
    output:
        f"{RESULTS}/conceptual_constants.tex",
    shell:
        "python {input} --output {output}"


rule main_result_figures:
    input:
        data=f"{SUMMARY_DIR}/instance_metrics.csv",
        script="scripts/plotting/plot_main_summary.py",
        families="scripts/plotting/family_summary.py",
        style="afabench/plotting/methods.py",
    output:
        expand(
            f"{RESULTS}/{{name}}.pdf",
            name=["main_summary_absolute_mcar", "main_summary_absolute_grid",
                  "main_summary_variants_grid", "law_grid"],
        ),
        f"{RESULTS}/main_summary.variant_cells.csv",
        f"{RESULTS}/main_summary.family_instances.csv",
        f"{RESULTS}/main_summary.family_cells.csv",
        f"{RESULTS}/main_summary.cells.csv",
    shell:
        "python {input.script} --summary-root {ROOT}/summary/val "
        "--output-dir {RESULTS}"


rule state_conditioning_figure:
    input:
        data=f"{SUMMARY_DIR}/instance_metrics.csv",
        script="scripts/plotting/plot_state_conditioning.py",
    output:
        f"{RESULTS}/state_conditioning.pdf",
    shell:
        "python {input.script} --summary-root {ROOT}/summary/val "
        "--mechanism all --output {output}"


rule route_structure:
    input:
        classifiers=ROUTE_CLASSIFIERS,
        data=f"{SUMMARY_DIR}/instance_metrics.csv",
        script="scripts/analysis/route_redundancy.py",
    output:
        f"{ROOT}/analysis/route_redundancy_{NAMESPACE}_val.csv",
    shell:
        "python {input.script} --namespace {NAMESPACE} "
        "--selection-split train --split val --k 2000 --seed 0 --device cpu"


rule route_structure_table:
    input:
        data=f"{ROOT}/analysis/route_redundancy_{NAMESPACE}_val.csv",
        script="scripts/analysis/route_structure_table.py",
    output:
        tex=f"{RESULTS}/route_structure.tex",
        csv=f"{RESULTS}/route_structure.csv",
    shell:
        "python {input.script} --output {output.tex} --csv-output {output.csv}"


rule collect_study_compute:
    input:
        COMPUTE_TRIGGERS,
    output:
        f"{RESULTS}/compute.csv",
    shell:
        "python scripts/analysis/collect_compute.py "
        "--namespace {NAMESPACE} --output {output}"


rule compute_figure:
    input:
        data=f"{RESULTS}/compute.csv",
        metrics=f"{SUMMARY_DIR}/instance_metrics.csv",
        script="scripts/plotting/plot_compute.py",
    output:
        f"{RESULTS}/compute.pdf",
    shell:
        "python {input.script} --compute {input.data} --output {output}"


rule plot_missing_data:
    input:
        script="scripts/plotting/plot_missing_data.py",
        style="extra/conf/scripts/plotting/common/default.yaml",
        study_style="extra/conf/scripts/plotting/common/missing_data.yaml",
        instances=f"{SUMMARY_DIR}/instance_metrics.csv",
        summary=f"{SUMMARY_DIR}/summary.csv",
        actions=f"{SUMMARY_DIR}/action_rates.csv",
        restoration=f"{SUMMARY_DIR}/restoration_rmse.csv",
    output:
        directory(FIGURE_DIR),
    resources:
        shell_exec="bash"
    shell:
        """
        python {input.script} \
            instance_metrics={input.instances} summary={input.summary} \
            action_rates={input.actions} restoration_rmse={input.restoration} \
            output_folder={output} formats='[pdf,svg]' \
            {HYDRA_WORKFLOW_OVERRIDES}
        """
