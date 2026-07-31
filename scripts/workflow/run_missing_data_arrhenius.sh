#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <missing-data-profile> [snakemake arguments...]" >&2
    exit 2
fi

profile=$1
shift
profile_dir="extra/workflow/profiles/config/${profile}"

source ~/.bashrc.d/afabench.sh
cd "${AFA_BASE}/repo/AFA-Benchmark"

if [[ ! -d ${profile_dir} ]]; then
    echo "unknown profile: ${profile}" >&2
    exit 2
fi

: "${SLURM_JOB_ID:?submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?request CPUs with sbatch --cpus-per-task}"
: "${SNIC_TMP:?Arrhenius did not provide job-local scratch}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# All Snakemake rules share this allocation. Give each shell invocation its own
# node-local directory so concurrent Hydra and Lightning jobs never collide.
# Arrhenius removes the allocation's outer SNIC_TMP when the job ends.
export AFABENCH_SHELL_PREFIX='source "$AFA_BASE"/venvs/$(uname -m)/bin/activate; export SNIC_TMP=$(mktemp -d "$SNIC_TMP/afabench-job.XXXXXX"); export TMPDIR="$SNIC_TMP"; '

exec uv run snakemake \
    --profile "${profile_dir}" all \
    --cores "${SLURM_CPUS_PER_TASK}" \
    --config device=cpu \
    --rerun-incomplete \
    --printshellcmds \
    "$@"
