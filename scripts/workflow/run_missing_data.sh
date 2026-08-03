#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: run_missing_data.sh --profile NAME --device DEVICE [options] [-- SNAKEMAKE_ARGS...]

options:
  --cores N        concurrent CPU slots (default: allocation CPUs or 4)
  --mem-mb N       schedulable memory in MiB (default: 90% of allocation or 22000)
  --gpu-slots N    concurrent CUDA rules (default: 1 for CUDA, otherwise 0)

The same command runs locally or inside one Slurm allocation. CUDA rules each
consume one `gpu` resource, so --gpu-slots 1 prevents GPU oversubscription while
independent CPU-only rules can still overlap.
EOF
    exit 2
}

profile=""
device=""
cores=""
mem_mb=""
gpu_slots=""
snakemake_args=()

while (($#)); do
    case "$1" in
        --profile) profile=${2:?}; shift 2 ;;
        --device) device=${2:?}; shift 2 ;;
        --cores) cores=${2:?}; shift 2 ;;
        --mem-mb) mem_mb=${2:?}; shift 2 ;;
        --gpu-slots) gpu_slots=${2:?}; shift 2 ;;
        --) shift; snakemake_args=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n ${profile} && -n ${device} ]] || usage
[[ ${cores:-${SLURM_CPUS_PER_TASK:-4}} =~ ^[1-9][0-9]*$ ]] || {
    echo "--cores must be a positive integer" >&2
    exit 2
}
cores=${cores:-${SLURM_CPUS_PER_TASK:-4}}

if [[ -z ${mem_mb} ]]; then
    if [[ -n ${SLURM_MEM_PER_NODE:-} ]]; then
        mem_mb=$((SLURM_MEM_PER_NODE * 9 / 10))
    else
        mem_mb=22000
    fi
fi
[[ ${mem_mb} =~ ^[1-9][0-9]*$ ]] || {
    echo "--mem-mb must be a positive integer" >&2
    exit 2
}

if [[ -z ${gpu_slots} ]]; then
    gpu_slots=0
    [[ ${device} == cuda* ]] && gpu_slots=1
fi
[[ ${gpu_slots} =~ ^[0-9]+$ ]] || {
    echo "--gpu-slots must be a non-negative integer" >&2
    exit 2
}
if [[ ${device} == cuda* && ${gpu_slots} -eq 0 ]]; then
    echo "CUDA execution requires at least one GPU slot" >&2
    exit 2
fi

repo_root=${AFABENCH_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
profile_dir="extra/workflow/profiles/config/${profile}"
cd "${repo_root}"
[[ -d ${profile_dir} ]] || {
    echo "unknown profile: ${profile}" >&2
    exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export AFABENCH_DEVICE=${device}

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    # Arrhenius keeps one environment per architecture on project storage.
    # Sourcing inside the allocation selects aarch64 on GH200 and x86_64 on
    # CPU nodes.
    if [[ -r ${HOME}/.bashrc.d/afabench.sh ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.bashrc.d/afabench.sh"
    fi
    : "${SNIC_TMP:?Arrhenius did not provide job-local scratch}"
    export AFABENCH_SCRATCH_ROOT=${SNIC_TMP}
    # A fresh directory per shell command prevents concurrent Hydra and
    # Lightning processes from sharing logs or checkpoints.
    export AFABENCH_SHELL_PREFIX='rule_tmp=$(mktemp -d "$AFABENCH_SCRATCH_ROOT/afabench-rule.XXXXXX"); export SNIC_TMP="$rule_tmp"; export TMPDIR="$rule_tmp"; '
else
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/afabench-uv-cache}
fi

if [[ ${device} == cuda* ]]; then
    uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this environment"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
fi

run_id=${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}
dry_run=false
for argument in "${snakemake_args[@]}"; do
    [[ ${argument} == -n || ${argument} == --dry-run ]] && dry_run=true
done

if [[ ${dry_run} == false ]]; then
    uv run python scripts/workflow/write_run_manifest.py \
        --profile "${profile}" --run-id "${run_id}" --device "${device}" \
        --cores "${cores}" --mem-mb "${mem_mb}" \
        --gpu-slots "${gpu_slots}" \
        --snakemake-args "${snakemake_args[@]}"
fi

exec uv run snakemake \
    --profile "${profile_dir}" all \
    --apptainer-prefix .snakemake/apptainer \
    --cores "${cores}" \
    --resources "mem_mb=${mem_mb}" "gpu=${gpu_slots}" \
    --default-resources mem_mb=3000 gpu=0 \
    --set-resources eval_missing_data_method:mem_mb=12000 \
    --rerun-incomplete \
    --printshellcmds \
    "${snakemake_args[@]}"
