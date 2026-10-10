#!/bin/bash
# Run a command inside the AFABench image for this node's architecture.
#
#   containers/run.sh python scripts/train_classifier/<script>.py ...
#
# Picks <image_dir>/afabench-<arch>.sif, where <image_dir> is $AFABENCH_IMAGE_DIR
# or this directory, binds the checkout this script belongs to and runs the
# command from it with the checkout on PYTHONPATH. GPUs are passed through
# when the node has any.

set -euo pipefail

# Physical paths, because a checkout reached through a symlink (Arrhenius's
# ~/projects -> /nobackup) is only visible in the image at its real location.
checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="${AFABENCH_IMAGE_DIR:-$checkout/containers}/afabench-$(uname -m).sif"
if [[ ! -f "$image" ]]; then
    echo "No image for $(uname -m) at $image; build it with containers/build.sbatch (docs/how-to/slurm_integration.md, \"Run jobs in an image\")" >&2
    exit 1
fi

gpu_flags=()
if command -v nvidia-smi > /dev/null 2>&1; then
    gpu_flags=(--nv)
fi

exec apptainer exec "${gpu_flags[@]}" --bind "$checkout" --pwd "$(pwd -P)" \
    --env "PYTHONPATH=$checkout${PYTHONPATH:+:$PYTHONPATH}" \
    "$image" "$@"
