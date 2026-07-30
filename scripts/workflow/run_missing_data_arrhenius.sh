#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <missing-data-profile>" >&2
    exit 2
fi

profile=$1
profile_dir="extra/workflow/profiles/config/${profile}"

source ~/.bashrc.d/afabench.sh
cd "${AFA_BASE}/repo/AFA-Benchmark"

if [[ ! -d ${profile_dir} ]]; then
    echo "unknown profile: ${profile}" >&2
    exit 2
fi

export AFABENCH_SHELL_PREFIX='source "$AFA_BASE"/venvs/$(uname -m)/bin/activate; '

exec uv run snakemake \
    --profile "${profile_dir}" all \
    --workflow-profile extra/workflow/profiles/arrhenius \
    --cores 200 \
    --rerun-incomplete \
    --printshellcmds
